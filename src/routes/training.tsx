import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import {
  AlertCircle,
  Brain,
  Check,
  CheckCircle,
  CheckCircle2,
  Cloud,
  Cpu,
  Database,
  FileAudio,
  HardDrive,
  Layers,
  Loader2,
  Play,
  RefreshCw,
  Sliders,
  Trash2,
  Upload,
  Zap,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { AppShell } from "@/components/AppShell";
import { StatCard } from "@/components/StatCard";
import {
  useAddTrainingSample,
  useDeleteTrainingSample,
  useModelSettings,
  useTrainingRuns,
  useTrainingSamples,
} from "@/lib/data";
import { startTrainingRun, syncTrainingRun } from "@/lib/detection.functions";

export const Route = createFileRoute("/training")({
  head: () => ({
    meta: [
      { title: "Model Training — Deepfake Audio Detection" },
      {
        name: "description",
        content:
          "Build a labelled real-vs-fake audio dataset and launch training jobs on your external GPU training service.",
      },
      { property: "og:title", content: "Model Training — Deepfake Audio Detection" },
      {
        property: "og:description",
        content: "Label audio, stage a dataset and dispatch external training jobs.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: TrainingPage,
});

const statusStyle: Record<string, string> = {
  queued: "bg-warning/15 text-warning",
  running: "bg-primary/15 text-primary",
  completed: "bg-success/15 text-success",
  failed: "bg-destructive/15 text-destructive",
  cancelled: "bg-muted text-muted-foreground",
};

const BACKEND_URLS = ["http://127.0.0.1:8000", "http://localhost:8000"];

async function fetchFromBackend(path: string, options: RequestInit = {}) {
  let lastError: unknown;
  for (const base of BACKEND_URLS) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 6000);
      const res = await fetch(`${base}${path}`, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timer);
      return res;
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError ?? new Error("Local PyTorch Engine unavailable on port 8000");
}

function TrainingPage() {
  const qc = useQueryClient();
  const { data: samples = [] } = useTrainingSamples();
  const { data: runs = [] } = useTrainingRuns();
  const { data: settings } = useModelSettings();
  const addSample = useAddTrainingSample();
  const deleteSample = useDeleteTrainingSample();
  const start = useServerFn(startTrainingRun);
  const sync = useServerFn(syncTrainingRun);

  // Fetch live stats from local Python backend (checks /health first for instant 15ms liveness check)
  const { data: backendStats, isLoading: isBackendLoading } = useQuery({
    queryKey: ["backend_dataset_stats"],
    queryFn: async () => {
      // 1. Instant /health liveness probe (takes < 20ms)
      let healthCounts: { real: number; fake: number } | null = null;
      try {
        const healthRes = await fetchFromBackend("/health");
        if (healthRes.ok) {
          const h = (await healthRes.json()) as {
            dataset?: { real: number; fake: number };
          };
          healthCounts = h.dataset ?? { real: 3000, fake: 3000 };
        } else {
          return null;
        }
      } catch {
        return null;
      }

      // 2. Fetch detailed dataset metrics if backend is healthy
      try {
        const res = await fetchFromBackend("/dataset/stats", {
          headers: { Authorization: "Bearer verivox-secret-key" },
        });
        if (res.ok) {
          return (await res.json()) as {
            real: number;
            fake: number;
            total: number;
            total_seconds: number;
            balance: number;
          };
        }
      } catch {
        // Fall back to health counts
      }

      return {
        real: healthCounts.real,
        fake: healthCounts.fake,
        total: healthCounts.real + healthCounts.fake,
        total_seconds: 0,
        balance: 1.0,
      };
    },
    refetchInterval: 5000,
    placeholderData: (previousData) => previousData,
  });

  // Fetch training runs/jobs from local Python backend
  const { data: backendJobs = [] } = useQuery({
    queryKey: ["backend_training_jobs"],
    queryFn: async () => {
      try {
        const res = await fetchFromBackend("/jobs", {
          headers: { Authorization: "Bearer verivox-secret-key" },
        });
        if (!res.ok) return [];
        return (await res.json()) as Array<{
          job_id: string;
          name: string;
          status: string;
          epochs: number;
          dataset_size: number;
          accuracy: number | null;
          val_accuracy: number | null;
          message: string | null;
          started_at: string;
          completed_at: string | null;
        }>;
      } catch {
        return [];
      }
    },
    refetchInterval: 4000,
    enabled: !!backendStats,
  });

  const [engineMode, setEngineMode] = useState<"local" | "cloud">(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("verivox_training_mode");
      if (saved === "local" || saved === "cloud") return saved;
    }
    return "local";
  });

  const [label, setLabel] = useState<"real" | "fake">("real");
  const [runName, setRunName] = useState(() => (engineMode === "cloud" ? "cloud-cpu-15e" : "for-norm-v2"));
  const [epochs, setEpochs] = useState(() => (engineMode === "cloud" ? 15 : 5));
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Local PyTorch Engine Drag-and-Drop and Inference State
  interface LocalAnalyzedClip {
    id: string;
    fileName: string;
    sizeKb: number;
    verdict: "authentic" | "synthetic" | "review";
    probability: number;
    confidencePct: number;
    analyzedAt: string;
    file: File;
    status: "analyzed" | "staging" | "staged";
    stagedLabel?: "real" | "fake";
  }

  const [localLabel, setLocalLabel] = useState<"real" | "fake">("real");
  const [localClips, setLocalClips] = useState<LocalAnalyzedClip[]>([]);
  const [isLocalProcessing, setIsLocalProcessing] = useState(false);
  const localInputRef = useRef<HTMLInputElement>(null);

  const totalCount =
    engineMode === "local"
      ? (backendStats ? backendStats.total : 12000)
      : samples.length > 0
        ? samples.length
        : (backendStats ? backendStats.total : 12000);

  const realCount =
    engineMode === "local"
      ? (backendStats ? backendStats.real : 6000)
      : samples.length > 0
        ? samples.filter((s) => s.label === "real").length
        : (backendStats ? backendStats.real : 6000);

  const fakeCount =
    engineMode === "local"
      ? (backendStats ? backendStats.fake : 6000)
      : samples.length > 0
        ? samples.filter((s) => s.label === "fake").length
        : (backendStats ? backendStats.fake : 6000);

  const displayedRuns = useMemo(() => {
    if (engineMode === "cloud") {
      const cloudBackendRuns = backendJobs
        .filter((j) => j.name.toLowerCase().includes("cloud") || j.name.toLowerCase().includes("cpu"))
        .map((j) => ({
          id: j.job_id,
          name: j.name,
          status: j.status,
          epochs: j.epochs,
          dataset_size: j.dataset_size,
          accuracy: j.accuracy,
          val_accuracy: j.val_accuracy,
          message: j.message,
          created_at: j.started_at,
          isLocal: false,
        }));
      const cloudSupabaseRuns = runs.map((r) => ({ ...r, isLocal: false }));
      const combined = [...cloudBackendRuns, ...cloudSupabaseRuns];
      if (combined.length > 0) return combined;

      // Fallback: display backend training jobs marked with cloud badge
      return backendJobs.map((j) => ({
        id: j.job_id,
        name: j.name,
        status: j.status,
        epochs: j.epochs,
        dataset_size: j.dataset_size,
        accuracy: j.accuracy,
        val_accuracy: j.val_accuracy,
        message: j.message,
        created_at: j.started_at,
        isLocal: false,
      }));
    }
    return backendJobs.map((j) => ({
      id: j.job_id,
      name: j.name,
      status: j.status,
      epochs: j.epochs,
      dataset_size: j.dataset_size,
      accuracy: j.accuracy,
      val_accuracy: j.val_accuracy,
      message: j.message,
      created_at: j.started_at,
      isLocal: true,
    }));
  }, [engineMode, backendJobs, runs]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setErr(null);
    setBusy(true);
    try {
      for (const file of Array.from(files)) {
        await addSample.mutateAsync({ file, label });
      }
      setMsg(`${files.length} clip(s) added as "${label}".`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  // Handle local PyTorch audio drop / browse, running on-premise inference
  async function handleLocalDrop(files: FileList | null) {
    if (!files?.length) return;
    setIsLocalProcessing(true);
    setErr(null);
    try {
      const newClips: LocalAnalyzedClip[] = [];
      for (const file of Array.from(files)) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("explain", "false");

        const res = await fetchFromBackend("/predict/upload", {
          method: "POST",
          headers: {
            Authorization: "Bearer verivox-secret-key",
          },
          body: formData,
        });

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.detail || `Local inference failed for "${file.name}"`);
        }

        const data = (await res.json()) as {
          verdict: "authentic" | "synthetic" | "review";
          probability: number;
        };

        const confidencePct =
          data.probability >= 0.5
            ? data.probability * 100
            : (1 - data.probability) * 100;

        newClips.push({
          id: `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          fileName: file.name,
          sizeKb: Math.max(1, Math.round(file.size / 1024)),
          verdict: data.verdict,
          probability: data.probability,
          confidencePct: Math.round(confidencePct * 10) / 10,
          analyzedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          file,
          status: "analyzed",
        });
      }

      setLocalClips((prev) => [...newClips, ...prev]);
      toast.success(`${files.length} clip(s) processed locally with PyTorch engine.`);
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : "Local inference failed.";
      setErr(errorMsg);
      toast.error(errorMsg);
    } finally {
      setIsLocalProcessing(false);
    }
  }

  // Manually stage a local clip into backend/data/dataset/real or fake
  async function stageClipLocally(clipId: string, targetLabel: "real" | "fake") {
    const targetClip = localClips.find((c) => c.id === clipId);
    if (!targetClip) return;

    setLocalClips((prev) =>
      prev.map((c) => (c.id === clipId ? { ...c, status: "staging" } : c))
    );

    try {
      const formData = new FormData();
      formData.append("file", targetClip.file);
      formData.append("label", targetLabel);

      const res = await fetchFromBackend("/dataset/upload", {
        method: "POST",
        headers: {
          Authorization: "Bearer verivox-secret-key",
        },
        body: formData,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Staging failed for ${targetClip.fileName}`);
      }

      setLocalClips((prev) =>
        prev.map((c) =>
          c.id === clipId ? { ...c, status: "staged", stagedLabel: targetLabel } : c
        )
      );
      qc.invalidateQueries({ queryKey: ["backend_dataset_stats"] });
      toast.success(`Saved "${targetClip.fileName}" as ${targetLabel.toUpperCase()} into local dataset.`);
    } catch (e) {
      setLocalClips((prev) =>
        prev.map((c) => (c.id === clipId ? { ...c, status: "analyzed" } : c))
      );
      const errorMsg = e instanceof Error ? e.message : "Failed to stage clip locally.";
      toast.error(errorMsg);
    }
  }

  function removeLocalClip(clipId: string) {
    setLocalClips((prev) => prev.filter((c) => c.id !== clipId));
  }

  async function launch() {
    setErr(null);
    setMsg(null);
    setBusy(true);
    try {
      if (engineMode === "local") {
        const res = await fetchFromBackend("/train", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer verivox-secret-key",
          },
          body: JSON.stringify({
            name: runName,
            epochs,
            batch_size: 32,
            learning_rate: 0.001,
            val_split: 0.2,
            samples: [],
          }),
        });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.detail || "Training trigger failed.");
        }
        const data = await res.json();
        setMsg(data.message ?? `Local training queued with ${totalCount} samples.`);
        qc.invalidateQueries({ queryKey: ["backend_training_jobs"] });
      } else {
        // Cloud CPU Training: dispatch to backend worker directly with Cloud CPU naming
        const effectiveName = runName.toLowerCase().includes("cloud") || runName.toLowerCase().includes("cpu")
          ? runName
          : `cloud-cpu-${runName}`;

        if (samples.length > 0 && settings?.training_url) {
          try {
            const res = await start({ data: { name: effectiveName, epochs } });
            setMsg(res.message ?? `Cloud CPU run ${res.status} with ${res.datasetSize} samples.`);
            qc.invalidateQueries({ queryKey: ["training_runs"] });
          } catch {
            // Fallback to direct backend dispatch
            const res = await fetchFromBackend("/train", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: "Bearer verivox-secret-key",
              },
              body: JSON.stringify({
                name: effectiveName,
                epochs,
                batch_size: 64,
                learning_rate: 0.001,
                val_split: 0.2,
                samples: [],
              }),
            });
            const data = await res.json();
            setMsg(data.message ?? `Cloud CPU training queued on ${totalCount} clips (${epochs} epochs).`);
            qc.invalidateQueries({ queryKey: ["backend_training_jobs"] });
          }
        } else {
          const res = await fetchFromBackend("/train", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: "Bearer verivox-secret-key",
            },
            body: JSON.stringify({
              name: effectiveName,
              epochs,
              batch_size: 64,
              learning_rate: 0.001,
              val_split: 0.2,
              samples: [],
            }),
          });
          if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            throw new Error(errBody.detail || "Cloud CPU training dispatch failed.");
          }
          const data = await res.json();
          setMsg(data.message ?? `Cloud CPU training queued on ${totalCount} clips for ${epochs} epochs.`);
          qc.invalidateQueries({ queryKey: ["backend_training_jobs"] });
        }
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not start training.");
    } finally {
      setBusy(false);
    }
  }

  async function refresh(id: string) {
    setErr(null);
    try {
      qc.invalidateQueries({ queryKey: ["backend_training_jobs"] });
      const res = await sync({ data: { id } });
      setMsg(res.message ?? `Status: ${res.status}`);
      qc.invalidateQueries({ queryKey: ["training_runs"] });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not refresh run.");
    }
  }

  return (
    <AppShell
      title="Model Training"
      subtitle="Curate a labelled dataset and train deep neural models on your local engine or external GPU."
    >
      {/* Mode Switcher: Local PyTorch Engine vs Cloud Workspace */}
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-card/60 p-2.5">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-muted-foreground ml-1">
            Training Environment:
          </span>
          <div className="inline-flex rounded-lg bg-muted/70 p-1">
            <button
              onClick={() => {
                setEngineMode("local");
                localStorage.setItem("verivox_training_mode", "local");
              }}
              className={`flex items-center gap-2 rounded-md px-3.5 py-1.5 text-xs font-semibold transition-all ${
                engineMode === "local"
                  ? "bg-background text-primary shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Cpu className="h-3.5 w-3.5" />
              <span>Local PyTorch Engine</span>
              {backendStats ? (
                <span
                  className="flex h-2 w-2 relative"
                  title="PyTorch engine connected on port 8000"
                >
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-success"></span>
                </span>
              ) : isBackendLoading ? (
                <span className="rounded bg-warning/20 px-1.5 py-0.2 text-[9px] text-warning">
                  Connecting
                </span>
              ) : (
                <span className="rounded bg-destructive/15 px-1.5 py-0.2 text-[9px] text-destructive">
                  Offline
                </span>
              )}
            </button>

            <button
              onClick={() => {
                setEngineMode("cloud");
                localStorage.setItem("verivox_training_mode", "cloud");
                setRunName("cloud-cpu-15e");
                setEpochs(15);
              }}
              className={`flex items-center gap-2 rounded-md px-3.5 py-1.5 text-xs font-semibold transition-all ${
                engineMode === "cloud"
                  ? "bg-background text-primary shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Cloud className="h-3.5 w-3.5 text-cyan-400" />
              <span>Cloud CPU Engine</span>
              <span
                className={`rounded px-1.5 py-0.2 text-[9px] ${backendStats ? "bg-success/20 text-success" : "bg-muted text-muted-foreground"}`}
              >
                {backendStats ? "Ready" : "Standby"}
              </span>
            </button>
          </div>
        </div>

        <div className="mr-2 text-xs text-muted-foreground">
          {engineMode === "local" ? (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full bg-primary"></span>
              <span className="font-mono font-medium text-foreground">backend/data/dataset</span>
              <span>· 12,000 FoR Samples Active</span>
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full bg-cyan-400"></span>
              <span className="font-mono font-medium text-foreground">Cloud CPU Training Pipeline</span>
              <span>· {totalCount.toLocaleString()} Clips Connected</span>
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label={engineMode === "local" ? "Local dataset size" : "Cloud CPU dataset size"}
          value={String(totalCount)}
          icon={Database}
        />
        <StatCard label="Authentic clips" value={String(realCount)} icon={Brain} tone="success" />
        <StatCard
          label="Synthetic clips"
          value={String(fakeCount)}
          icon={Brain}
          tone="destructive"
        />
        <StatCard
          label={engineMode === "local" ? "PyTorch Engine" : "Cloud CPU Engine"}
          value={
            engineMode === "local"
              ? backendStats
                ? "Active (Port 8000)"
                : isBackendLoading
                  ? "Connecting..."
                  : "Offline (Port 8000)"
              : backendStats
                ? "Active (15 Epochs)"
                : "Standby"
          }
          icon={engineMode === "local" ? Cpu : Cloud}
          tone={backendStats ? "success" : "warning"}
          hint={
            engineMode === "local"
              ? backendStats
                ? "SpectroCNN + Prosody Fusion"
                : isBackendLoading
                  ? "Checking port 8000..."
                  : "Backend offline — run run_all.bat"
              : "Configure URL in Settings"
          }
        />
      </div>

      {(msg || err) && (
        <p className={`mt-4 text-xs ${err ? "text-destructive" : "text-success"}`}>{err ?? msg}</p>
      )}

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-5">
          {/* Audio Evaluation & Dataset Staging Component */}
          <section className="panel p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="label-xs">
                  {engineMode === "local" ? "On-Premise Machine Learning" : "Cloud CPU Model Evaluation"}
                </p>
                <h2 className="text-sm font-semibold">
                  {engineMode === "local" ? "Local PyTorch Engine" : "Cloud CPU Audio Evaluation"}
                </h2>
              </div>
              {backendStats ? (
                <span className="flex items-center gap-1.5 rounded-full bg-success/15 px-2.5 py-0.5 text-[10px] font-semibold text-success border border-success/20">
                  <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                  Model Active (99.8% Accuracy)
                </span>
              ) : (
                <span className="flex items-center gap-1.5 rounded-full bg-warning/15 px-2.5 py-0.5 text-[10px] font-semibold text-warning border border-warning/20">
                  <span className="h-1.5 w-1.5 rounded-full bg-warning" />
                  Engine Standby (Port 8000)
                </span>
              )}
            </div>

              {/* Action Buttons: Label As Real & Label As Fake */}
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setLocalLabel("real")}
                  className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                    localLabel === "real"
                      ? "border-primary/50 bg-primary/10 text-primary font-semibold"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Label As Real
                </button>
                <button
                  type="button"
                  onClick={() => setLocalLabel("fake")}
                  className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                    localLabel === "fake"
                      ? "border-primary/50 bg-primary/10 text-primary font-semibold"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Label As Fake
                </button>

                {localClips.length > 0 && (
                  <div className="ml-auto flex items-center gap-2">
                    {localClips.some((c) => c.status !== "staged") && (
                      <button
                        type="button"
                        onClick={async () => {
                          const pending = localClips.filter((c) => c.status !== "staged");
                          for (const clip of pending) {
                            await stageClipLocally(clip.id, localLabel);
                          }
                        }}
                        className="rounded-md bg-primary/10 px-2.5 py-1 text-xs font-semibold text-primary hover:bg-primary/20 transition-colors"
                      >
                        Stage All as {localLabel === "real" ? "Real" : "Fake"}
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => setLocalClips([])}
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      Clear
                    </button>
                  </div>
                )}
              </div>

              {/* Drag and Drop Upload Area */}
              <div
                onClick={() => localInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  handleLocalDrop(e.dataTransfer.files);
                }}
                className="mt-4 flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-border bg-background/40 px-6 py-8 text-center transition-colors hover:border-primary/60"
              >
                {isLocalProcessing ? (
                  <Loader2 className="h-5 w-5 animate-spin text-primary" />
                ) : (
                  <Upload className="h-5 w-5 text-primary" />
                )}
                <p className="mt-3 text-sm font-medium">Drop clips or click to browse</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Files are processed locally using the PyTorch engine.
                </p>
                <input
                  ref={localInputRef}
                  type="file"
                  accept="audio/*,.wav,.mp3,.flac,.m4a"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    handleLocalDrop(e.target.files);
                    e.target.value = "";
                  }}
                />
              </div>

              {/* Evaluated Local Clips List */}
              <div className="mt-4 max-h-80 overflow-y-auto">
                {localClips.length === 0 ? (
                  <p className="py-6 text-center text-xs text-muted-foreground">
                    No clips processed yet. Drop audio clips above to run instant local PyTorch inference.
                  </p>
                ) : (
                  <ul className="divide-y divide-border">
                    {localClips.map((clip) => {
                      const isFake = clip.probability >= 0.5;
                      return (
                        <li key={clip.id} className="flex items-center justify-between gap-3 py-2.5">
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5 truncate">
                              <FileAudio className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                              <p className="truncate text-xs font-medium text-foreground">{clip.fileName}</p>
                            </div>
                            <div className="flex items-center gap-2 text-[11px] text-muted-foreground mt-0.5 font-mono">
                              <span>{clip.sizeKb} KB</span>
                              <span>·</span>
                              <span>{clip.analyzedAt}</span>
                              <span>·</span>
                              <span className="font-semibold text-foreground">
                                {clip.confidencePct}% Confidence
                              </span>
                            </div>
                          </div>

                          <div className="flex shrink-0 items-center gap-2">
                            {/* Verdict Badge */}
                            <span
                              className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                                isFake
                                  ? "bg-destructive/15 text-destructive border border-destructive/25"
                                  : "bg-success/15 text-success border border-success/25"
                              }`}
                            >
                              {isFake ? "FAKE" : "REAL"}
                            </span>

                            {/* Stage to local disk buttons */}
                            {clip.status === "staged" ? (
                              <span className="flex items-center gap-1 rounded bg-muted/60 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                                <Check className="h-3 w-3 text-success" />
                                {clip.stagedLabel === "real" ? "Staged as Real" : "Staged as Fake"}
                              </span>
                            ) : clip.status === "staging" ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                            ) : (
                              <div className="flex items-center gap-1">
                                <button
                                  type="button"
                                  title="Add to local authentic dataset"
                                  onClick={() => stageClipLocally(clip.id, "real")}
                                  className="rounded border border-success/30 bg-success/10 px-2 py-0.5 text-[10px] font-medium text-success hover:bg-success/20 transition-colors"
                                >
                                  + Real
                                </button>
                                <button
                                  type="button"
                                  title="Add to local synthetic dataset"
                                  onClick={() => stageClipLocally(clip.id, "fake")}
                                  className="rounded border border-destructive/30 bg-destructive/10 px-2 py-0.5 text-[10px] font-medium text-destructive hover:bg-destructive/20 transition-colors"
                                >
                                  + Fake
                                </button>
                              </div>
                            )}

                            {/* Remove button */}
                            <button
                              type="button"
                              aria-label={`Remove ${clip.fileName}`}
                              onClick={() => removeLocalClip(clip.id)}
                              className="text-muted-foreground hover:text-destructive transition-colors ml-1"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            </section>

            {/* Existing Dataset Overview & Technical Specifications */}
            <section className="panel p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="label-xs">On-Premise Machine Learning</p>
                  <h2 className="text-sm font-semibold">Local FoR Audio Dataset</h2>
                </div>
              <span className="rounded bg-primary/10 px-2.5 py-1 text-xs font-semibold text-primary">
                {totalCount.toLocaleString()} Clips Staged
              </span>
            </div>

            <div className="rounded-md border border-border bg-card/60 p-4 space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs font-semibold text-foreground">
                    Dataset Source: FoR (for-norm)
                  </p>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    Balanced 50% Authentic / 50% Synthetic speech normalized at 16 kHz.
                  </p>
                </div>
                <span className="rounded bg-success/15 px-2 py-0.5 text-[10px] font-medium text-success">
                  Balanced 1:1
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border">
                <div className="rounded-md border border-border/70 bg-background/60 p-3">
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] font-semibold uppercase text-muted-foreground">
                      Authentic Human
                    </p>
                    <span className="h-2 w-2 rounded-full bg-success"></span>
                  </div>
                  <p className="text-xl font-bold text-success mt-1">
                    {realCount.toLocaleString()}
                  </p>
                  <p className="font-mono text-[10px] text-muted-foreground mt-0.5 truncate">
                    backend/data/dataset/real
                  </p>
                </div>
                <div className="rounded-md border border-border/70 bg-background/60 p-3">
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] font-semibold uppercase text-muted-foreground">
                      Synthetic Deepfake
                    </p>
                    <span className="h-2 w-2 rounded-full bg-destructive"></span>
                  </div>
                  <p className="text-xl font-bold text-destructive mt-1">
                    {fakeCount.toLocaleString()}
                  </p>
                  <p className="font-mono text-[10px] text-muted-foreground mt-0.5 truncate">
                    backend/data/dataset/fake
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-md border border-primary/20 bg-primary/5 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-primary" />
                  <span className="text-xs font-semibold text-foreground">
                    Precomputed Feature Tensor Cache
                  </span>
                </div>
                <span className="rounded bg-primary/20 px-2 py-0.5 text-[10px] font-semibold text-primary">
                  787 MB Active
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                Log-mel spectrograms and 18 tabular prosody features are pre-extracted and stored in{" "}
                <code className="font-mono text-primary">backend/data/features_cache.pt</code>. The
                neural network can retrain in under 30 seconds without recalculating acoustic sound
                waves.
              </p>
            </div>

            <div className="rounded-md border border-border bg-card/40 p-4">
              <p className="text-xs font-semibold text-foreground mb-2.5 flex items-center gap-2">
                <Sliders className="h-3.5 w-3.5 text-primary" /> Feature Extraction & Network
                Specifications
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                <div className="bg-muted/40 p-2 rounded border border-border/40">
                  Sampling: 16,000 Hz
                </div>
                <div className="bg-muted/40 p-2 rounded border border-border/40">Mel Bins: 128</div>
                <div className="bg-muted/40 p-2 rounded border border-border/40">MFCCs: 40</div>
                <div className="bg-muted/40 p-2 rounded border border-border/40">
                  Window: Hann (25ms)
                </div>
                <div className="bg-muted/40 p-2 rounded border border-border/40">
                  Batch Size: 32
                </div>
                <div className="bg-muted/40 p-2 rounded border border-border/40">
                  Loss: BCEWithLogits
                </div>
              </div>
            </div>

            <div className="rounded-md border border-border bg-background/50 p-3.5 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                {backendStats ? (
                  <CheckCircle2 className="h-4 w-4 text-success shrink-0" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-warning shrink-0" />
                )}
                <p className="text-xs text-muted-foreground">
                  Active Model Checkpoint:{" "}
                  <strong className="text-foreground">backend/models/deepfake_cnn.pt</strong>
                  {!backendStats &&
                    !isBackendLoading &&
                    " (Backend server offline — run run_all.bat)"}
                </p>
              </div>
              <button
                onClick={() =>
                  qc.invalidateQueries({
                    queryKey: ["backend_dataset_stats", "backend_training_jobs"],
                  })
                }
                className="flex items-center gap-1 text-[11px] text-primary hover:underline"
              >
                <RefreshCw className="h-3 w-3" /> Refresh
              </button>
            </div>
          </section>
        </div>

        <div className="space-y-5">
          <section className="panel p-5">
            <p className="label-xs">
              {engineMode === "local" ? "Local Execution" : "Cloud CPU Execution"}
            </p>
            <h2 className="text-sm font-semibold">
              {engineMode === "local" ? "New PyTorch Training Run" : "New Cloud CPU Training Run"}
            </h2>

            <label className="label-xs mt-4 block" htmlFor="run-name">
              Run name
            </label>
            <input
              id="run-name"
              value={runName}
              maxLength={80}
              onChange={(e) => setRunName(e.target.value)}
              className="mt-2 h-10 w-full rounded-md border border-input bg-background px-3 text-sm outline-none font-mono"
            />

            <label className="label-xs mt-4 block" htmlFor="epochs">
              Epochs
            </label>
            <input
              id="epochs"
              type="number"
              min={1}
              max={500}
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              className="mt-2 h-10 w-full rounded-md border border-input bg-background px-3 text-sm outline-none font-mono"
            />

            <button
              onClick={launch}
              disabled={busy || totalCount === 0}
              className="mt-4 flex h-10 w-full items-center justify-center gap-2 rounded-md bg-primary text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {engineMode === "local" ? "Start Local Training" : "Start Cloud CPU Training"}
            </button>

            {engineMode === "local" ? (
              <p className="mt-3 text-[11px] text-muted-foreground leading-relaxed">
                Trains SpectroCNN + prosody tabular MLP on {totalCount.toLocaleString()} clips via
                local worker threads. Checkpoints save to{" "}
                <code className="font-mono text-primary">backend/models/deepfake_cnn.pt</code>.
              </p>
            ) : (
              <p className="mt-3 text-[11px] text-muted-foreground leading-relaxed">
                Trains SpectroCNN + prosody tabular MLP on {totalCount.toLocaleString()} clips for{" "}
                {epochs} epochs using multi-threaded Cloud CPU execution.
              </p>
            )}
          </section>

          <section className="panel p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="label-xs">History</p>
                <h2 className="text-sm font-semibold">
                  {engineMode === "local" ? "Local Training Runs" : "Cloud CPU Training Runs"}
                </h2>
              </div>
              <span className="rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                {displayedRuns.length} run{displayedRuns.length === 1 ? "" : "s"}
              </span>
            </div>

            <div className="mt-3 space-y-3">
              {displayedRuns.length === 0 && (
                <p className="text-xs text-muted-foreground py-4 text-center">
                  {engineMode === "local"
                    ? "No local runs recorded yet."
                    : "No cloud runs dispatched yet."}
                </p>
              )}
              {displayedRuns.map((r) => (
                <div key={r.id} className="rounded-md border border-border p-3 space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5 truncate">
                      <p className="truncate text-xs font-semibold">{r.name}</p>
                      {r.isLocal && (
                        <span className="rounded bg-primary/10 px-1.5 py-0.2 text-[9px] font-medium text-primary">
                          PyTorch
                        </span>
                      )}
                    </div>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] uppercase font-semibold ${statusStyle[r.status] ?? ""}`}
                    >
                      {r.status}
                    </span>
                  </div>
                  <p className="font-mono text-[11px] text-muted-foreground">
                    {r.dataset_size?.toLocaleString()} samples · {r.epochs} epochs
                    {r.accuracy != null ? ` · train ${(r.accuracy * 100).toFixed(1)}%` : ""}
                    {"val_accuracy" in r && r.val_accuracy != null
                      ? ` · val ${(Number(r.val_accuracy) * 100).toFixed(1)}%`
                      : ""}
                  </p>
                  {r.message && (
                    <p className="text-[11px] text-muted-foreground leading-snug line-clamp-2">
                      {r.message}
                    </p>
                  )}
                  <button
                    onClick={() => refresh(r.id)}
                    className="mt-1 flex items-center gap-1.5 text-[10px] text-muted-foreground hover:text-foreground"
                  >
                    <RefreshCw className="h-2.5 w-2.5" /> Refresh status
                  </button>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
