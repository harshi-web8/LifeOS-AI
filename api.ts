import express, { Request, Response } from 'express';
import { GoogleGenAI } from '@google/genai';

export const apiRouter = express.Router();

apiRouter.use(express.json({ limit: '1mb' }));

const GEMINI_MODEL = 'gemini-3.1-flash-lite';

// Cache daily recommendations for 30 minutes to avoid unnecessary API calls on page switches
const RECS_CACHE_TTL_MS = 30 * 60 * 1000;
const recommendationsCache = new Map<string, { data: Record<string, any>; expiresAt: number }>();

const getGenAI = () => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    throw new Error('GEMINI_API_KEY is not configured in Settings > Secrets.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
};

const parseModelJson = <T>(rawText: string | undefined, fallback: T): T => {
  if (!rawText) return fallback;
  const cleaned = rawText
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  if (!cleaned) return fallback;
  return JSON.parse(cleaned) as T;
};

function getErrorStatus(err: any): number | undefined {
  if (typeof err?.status === 'number') return err.status;
  if (typeof err?.code === 'number') return err.code;
  const msg = String(err?.message || '');
  const match = msg.match(/"code"\s*:\s*(\d{3})/);
  return match ? Number(match[1]) : undefined;
}

function isQuotaExhaustedError(err: any): boolean {
  const status = getErrorStatus(err);
  const msg = String(err?.message || '');
  return status === 429 || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('Quota exceeded');
}

async function generateWithFallback<T>(
  _operationName: string,
  fn: (model: string) => Promise<T>
): Promise<{ result: T; modelUsed: string }> {
  try {
    const result = await fn(GEMINI_MODEL);
    return { result, modelUsed: GEMINI_MODEL };
  } catch (err: any) {
    const status = getErrorStatus(err);

    // Single quick retry only for transient 5xx server errors (never retry daily 429 quota exhaustion)
    if (typeof status === 'number' && status >= 500 && status < 600) {
      await new Promise((r) => setTimeout(r, 500));
      const retryResult = await fn(GEMINI_MODEL);
      return { result: retryResult, modelUsed: GEMINI_MODEL };
    }

    throw err;
  }
}

function formatGeminiErrorMessage(error: any, defaultMsg: string): string {
  const status = getErrorStatus(error);
  const raw = String(error?.message || defaultMsg);

  if (isQuotaExhaustedError(error)) {
    return 'Gemini API free-tier quota is temporarily exhausted. Please try again later or select a billing-enabled API key in Settings > Secrets.';
  }
  if (status === 403 || status === 400 || raw.includes('API_KEY_INVALID') || raw.includes('PERMISSION_DENIED')) {
    return 'Gemini API key is invalid or missing permissions. Please verify your API key in Settings > Secrets.';
  }
  return raw || defaultMsg;
}

function sendGeminiError(res: Response, error: any, defaultMsg: string) {
  const parsedStatus = getErrorStatus(error);
  const status =
    typeof parsedStatus === 'number' && parsedStatus >= 400 && parsedStatus < 600
      ? parsedStatus
      : 500;
  const friendlyMessage = formatGeminiErrorMessage(error, defaultMsg);

  if (status === 429) {
    console.warn(`Gemini API quota limit (${status}): ${friendlyMessage}`);
  } else {
    console.error(`Gemini API error (${status}):`, friendlyMessage);
  }

  return res.status(status).json({ error: friendlyMessage });
}

// Health check endpoint
apiRouter.get('/health', (_req: Request, res: Response) => {
  const hasKey = Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY');
  res.json({
    ok: true,
    model: GEMINI_MODEL,
    geminiConfigured: hasKey,
  });
});

// 1. Full-page AI Assistant Chat
apiRouter.post('/gemini/chat', async (req: Request, res: Response) => {
  try {
    const { message, history, userContext } = req.body;
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'Message is required' });
    }

    const ai = getGenAI();

    const systemInstruction = `You are LifeOS AI, an intelligent, empathetic, and sharp personal life intelligence assistant.
You have real-time access to the user's active life management data:
- Tasks: ${JSON.stringify(userContext?.tasks || [])}
- Timetable/Schedule: ${JSON.stringify(userContext?.timetable || [])}
- Study Goals & Deadlines: ${JSON.stringify(userContext?.studyItems || [])}
- Smart Notes: ${JSON.stringify(userContext?.notes || [])}
- Health & Fitness (steps, sleep, hydration, mood): ${JSON.stringify(userContext?.healthLogs || (userContext?.todayHealth ? [userContext.todayHealth] : []))}
- Habits: ${JSON.stringify(userContext?.habits || [])}
- Hobbies: ${JSON.stringify(userContext?.hobbies || [])}

Your capabilities:
1. Provide proactive, actionable productivity and wellness advice tailored to their specific data.
2. Suggest concrete timetable modifications and task priority adjustments.
3. Help reschedule missed tasks and balance work, study, health, and hobbies.
4. Keep answers concise, inspiring, well-structured with bullet points when helpful, without fluff or robotic filler.`;

    const chatHistory = Array.isArray(history)
      ? history
          .filter((item: any) => item && typeof item.content === 'string' && item.content.trim())
          .map((item: any) => ({
            role: item.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: item.content }],
          }))
      : [];

    const { result: response, modelUsed } = await generateWithFallback('chat', (model) =>
      ai.models.generateContent({
        model,
        contents: [...chatHistory, { role: 'user', parts: [{ text: message }] }],
        config: {
          systemInstruction,
        },
      })
    );

    const reply = response.text?.trim();
    if (!reply) {
      return res.status(502).json({ error: 'Gemini returned an empty response.' });
    }
    return res.json({ reply, modelUsed });
  } catch (error: any) {
    return sendGeminiError(res, error, 'Failed to generate AI response.');
  }
});

// 2. AI Timetable Generator
apiRouter.post('/gemini/timetable', async (req: Request, res: Response) => {
  try {
    const { subjects, wakeUpTime, sleepTime, studyHours, preferences, existingSlots } = req.body;
    const ai = getGenAI();

    const prompt = `Generate an optimized, balanced weekly timetable for Monday to Sunday.
Inputs:
- Subjects/Activities: ${JSON.stringify(subjects || ['Mathematics', 'Machine Learning', 'Strength Training', 'Reading', 'Coding Projects'])}
- Typical Wake Up: ${wakeUpTime || '07:00'}
- Typical Bedtime: ${sleepTime || '23:00'}
- Target Daily Focus Hours: ${studyHours || 4}
- Preferences: ${preferences || 'Include morning focus blocks, evening hobby time, and adequate breaks'}
- Existing fixed events: ${JSON.stringify(existingSlots || [])}

CRITICAL: Return ONLY valid JSON array of objects conforming strictly to this format:
[
  {
    "dayOfWeek": "Monday",
    "startTime": "09:00",
    "endTime": "10:30",
    "title": "Mathematics Calculus",
    "category": "Study",
    "color": "indigo"
  }
]
Do not return Markdown code blocks. Just the raw JSON.`;

    const { result: response, modelUsed } = await generateWithFallback('timetable', (model) =>
      ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
        },
      })
    );

    const slots = parseModelJson<any[]>(response.text, []);
    return res.json({ slots, modelUsed });
  } catch (error: any) {
    return sendGeminiError(res, error, 'Failed to generate timetable');
  }
});

// 3. AI Task Priority & Strategy Analyzer
apiRouter.post('/gemini/priorities', async (req: Request, res: Response) => {
  try {
    const { tasks } = req.body;
    const ai = getGenAI();

    const prompt = `Analyze these user tasks and optimize their priorities, recommended order of execution, and estimated focus time in minutes.
Tasks: ${JSON.stringify(tasks || [])}

CRITICAL: Return ONLY valid JSON conforming strictly to:
{
  "analyzedTasks": [
    {
      "id": "original-task-id",
      "recommendedPriority": "high",
      "estimatedMinutes": 45,
      "reason": "Upcoming deadline with highest study impact",
      "suggestedOrder": 1
    }
  ],
  "strategicAdvice": "Short 2-sentence summary of how the user should tackle their day."
}`;

    const { result: response, modelUsed } = await generateWithFallback('priorities', (model) =>
      ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
        },
      })
    );

    const parsed = parseModelJson<Record<string, any>>(response.text, {
      analyzedTasks: [],
      strategicAdvice: '',
    });
    return res.json({ ...parsed, modelUsed });
  } catch (error: any) {
    return sendGeminiError(res, error, 'Failed to analyze task priorities');
  }
});

// 4. AI Missed Task Rescheduler
apiRouter.post('/gemini/reschedule', async (req: Request, res: Response) => {
  try {
    const { missedTasks, timetableSlots } = req.body;
    const ai = getGenAI();

    const prompt = `The user missed the following tasks:
${JSON.stringify(missedTasks || [])}

Here is their current weekly timetable:
${JSON.stringify(timetableSlots || [])}

Find non-overlapping, realistic free slots over the next 3 days to reschedule these missed tasks without overloading the user.
Return ONLY valid JSON array:
[
  {
    "taskId": "task-id",
    "taskTitle": "Task title",
    "suggestedDay": "Tuesday",
    "startTime": "16:00",
    "endTime": "17:00",
    "reason": "Open 1-hour gap between afternoon study and gym"
  }
]`;

    const { result: response, modelUsed } = await generateWithFallback('reschedule', (model) =>
      ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
        },
      })
    );

    const suggestions = parseModelJson<any[]>(response.text, []);
    return res.json({ suggestions, modelUsed });
  } catch (error: any) {
    return sendGeminiError(res, error, 'Failed to reschedule tasks');
  }
});

// 5. Daily Life Intelligence Recommendations (with 30-min server cache)
apiRouter.post('/gemini/recommendations', async (req: Request, res: Response) => {
  try {
    const { userMetrics, forceRefresh } = req.body || {};
    const cacheKey = JSON.stringify(userMetrics || {});
    const now = Date.now();

    if (!forceRefresh) {
      const cached = recommendationsCache.get(cacheKey);
      if (cached && cached.expiresAt > now) {
        return res.json({ ...cached.data, cached: true });
      }
    }

    const ai = getGenAI();

    const prompt = `Based on the user's recent activity:
${JSON.stringify(userMetrics || {})}

Provide 3 high-impact, actionable life recommendations for today across Focus/Study, Physical Vitality/Health, and Mental Restoration/Hobbies.
Return JSON strictly in this structure:
{
  "focusRecommendation": {
    "title": "...",
    "action": "...",
    "expectedImpact": "..."
  },
  "vitalityRecommendation": {
    "title": "...",
    "action": "...",
    "expectedImpact": "..."
  },
  "restorationRecommendation": {
    "title": "...",
    "action": "...",
    "expectedImpact": "..."
  },
  "dailyMotto": "A punchy 8-word stoic or growth-mindset motto for today."
}`;

    const { result: response, modelUsed } = await generateWithFallback('recommendations', (model) =>
      ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
        },
      })
    );

    const parsed = parseModelJson<Record<string, any>>(response.text, {});
    if (parsed && Object.keys(parsed).length > 0) {
      if (recommendationsCache.size >= 200) {
        const oldestKey = recommendationsCache.keys().next().value;
        if (oldestKey) recommendationsCache.delete(oldestKey);
      }
      recommendationsCache.set(cacheKey, {
        data: parsed,
        expiresAt: now + RECS_CACHE_TTL_MS,
      });
    }
    return res.json({ ...parsed, modelUsed });
  } catch (error: any) {
    return sendGeminiError(res, error, 'Failed to generate recommendations');
  }
});

// 6. Smart Notes AI Enhancement & Summarization
apiRouter.post('/gemini/smart-note', async (req: Request, res: Response) => {
  try {
    const { title, content, mode } = req.body;
    const ai = getGenAI();

    const prompt = `You are the LifeOS Smart Notes Engine.
Note Title: ${title || 'Untitled'}
Note Content:
${content || ''}

Action requested: ${mode || 'enhance'} (Options: summarize, action_items, enhance, key_insights)

Please return ONLY a JSON response in this format:
{
  "enhancedTitle": "Cleaned up or improved title",
  "summary": "Concise 2-sentence executive summary",
  "keyPoints": ["Bullet point 1", "Bullet point 2"],
  "actionItems": ["Action item 1", "Action item 2"],
  "suggestedTags": ["tag1", "tag2"]
}`;

    const { result: response, modelUsed } = await generateWithFallback('smart-note', (model) =>
      ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
        },
      })
    );

    const parsed = parseModelJson<Record<string, any>>(response.text, {});
    return res.json({ ...parsed, modelUsed });
  } catch (error: any) {
    return sendGeminiError(res, error, 'Failed to enhance smart note');
  }
});
