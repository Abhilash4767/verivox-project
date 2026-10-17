// Server-only helpers for talking to an external inference / training service.

export interface InferenceExplanation {
  shap?: Array<{ feature: string; value: number; weight: number; direction: string }>;
  lime?: Array<{ feature: string; value: number; weight: number; direction: string }>;
  feature_importance?: Array<{ feature: string; value: number; weight: number; direction: string }>;
  spectrogram_heatmap?: number[][];
  saliency?: number[];
  narrative?: string;
}

export interface InferenceResult {
  probability: number;
  model?: string | undefined;
  durationSec?: number | undefined;
  sampleRate?: number | undefined;
  explanation?: InferenceExplanation | undefined;
}

function authHeaders(): Record<string, string> {
  const key = process.env["MODEL_API_KEY"] || "verivox-secret-key";
  return { Authorization: `Bearer ${key}` };
}

export async function callInference(
  url: string,
  body: Record<string, unknown>,
): Promise<InferenceResult> {
  const targetUrl = url.replace("//localhost:", "//127.0.0.1:");
  const res = await fetch(targetUrl, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(
      `Inference service responded ${res.status}: ${(await res.text()).slice(0, 300)}`,
    );
  }
  const json = (await res.json()) as Record<string, unknown>;
  const raw =
    (json["probability"] as number) ??
    (json["score"] as number) ??
    (json["fake_probability"] as number) ??
    (json["confidence"] as number);
  if (typeof raw !== "number" || Number.isNaN(raw)) {
    throw new Error("Inference service did not return a numeric probability/score field.");
  }
  return {
    probability: raw > 1 ? raw / 100 : raw,
    model: (json["model"] as string) ?? undefined,
    durationSec: (json["duration_sec"] as number) ?? undefined,
    sampleRate: (json["sample_rate"] as number) ?? undefined,
    explanation: (json["explanation"] as InferenceResult["explanation"]) ?? undefined,
  };
}

export async function callTraining(
  url: string,
  body: Record<string, unknown>,
): Promise<{ jobId: string | null; status: string; message?: string }> {
  const targetUrl = url.replace("//localhost:", "//127.0.0.1:");
  const res = await fetch(targetUrl, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`Training service responded ${res.status}: ${text.slice(0, 300)}`);
  let json: Record<string, unknown> = {};
  try {
    json = JSON.parse(text) as Record<string, unknown>;
  } catch {
    /* non-JSON response is tolerated */
  }
  return {
    jobId: (json["job_id"] as string) ?? (json["id"] as string) ?? null,
    status: (json["status"] as string) ?? "running",
    message: (json["message"] as string) ?? undefined,
  };
}

export async function pollTraining(url: string, jobId: string) {
  const base = url.replace("//localhost:", "//127.0.0.1:").replace(/\/+$/, "");
  const res = await fetch(`${base}/status/${encodeURIComponent(jobId)}`, {
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(`Training status responded ${res.status}`);
  return (await res.json()) as Record<string, unknown>;
}
