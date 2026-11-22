// QR code decoding and live camera scanning powered by zxing-wasm (ZXing-C++ in WebAssembly).
// The .wasm binary is bundled locally by Vite (?url) so scanning works offline and without a CDN.

import type { ReadResult, ReaderOptions } from "zxing-wasm/reader";
import zxingWasmUrl from "zxing-wasm/reader/zxing_reader.wasm?url";

type ZXingReaderMod = typeof import("zxing-wasm/reader");
type Facing = "environment" | "user";

let zxingPromise: Promise<ZXingReaderMod> | null = null;

async function getZXing(): Promise<ZXingReaderMod> {
  if (!zxingPromise) {
    zxingPromise = import("zxing-wasm/reader").then((mod) => {
      mod.prepareZXingModule({
        overrides: {
          locateFile: (path: string) => (path.endsWith(".wasm") ? zxingWasmUrl : path),
        },
      });
      return mod;
    }).catch((err) => {
      zxingPromise = null;
      throw err;
    });
  }
  return zxingPromise;
}

function extractText(results: ReadResult[]): string | null {
  for (const r of results) {
    if (!r.isValid) continue;
    if (r.text) return r.text;
    if (r.bytes?.length) return new TextDecoder().decode(r.bytes);
  }
  return null;
}

const BASE_OPTS: ReaderOptions = {
  formats: ["QRCode"],
  tryHarder: true,
  tryRotate: true,
  tryInvert: true,
  tryDownscale: true,
  maxNumberOfSymbols: 1,
  textMode: "Plain",
};

async function loadImageSource(f: File): Promise<{ src: CanvasImageSource; width: number; height: number; release: () => void }> {
  if (typeof createImageBitmap === "function") {
    try {
      const bmp = await createImageBitmap(f);
      return { src: bmp, width: bmp.width, height: bmp.height, release: () => bmp.close() };
    } catch { /* fall back to HTMLImageElement */ }
  }
  const url = URL.createObjectURL(f);
  const img = new Image();
  img.decoding = "async";
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("could not decode image"));
    img.src = url;
  });
  return { src: img, width: img.naturalWidth, height: img.naturalHeight, release: () => URL.revokeObjectURL(url) };
}

/** Decode a QR code from an uploaded image, phone photo, or pasted screenshot. */
export async function decodeQRFromFile(file: File): Promise<string> {
  const { readBarcodes } = await getZXing();

  // Pass 1: native C++ decoding directly on the raw file bytes (full resolution, no canvas loss).
  try {
    const direct = extractText(await readBarcodes(file, BASE_OPTS));
    if (direct) return direct;
  } catch { /* unsupported format for stb_image (e.g. WebP/HEIC): fall through to browser decode */ }

  // Pass 2: browser-decoded image on canvas with denoise, global binarizer, and centered crop.
  const loaded = await loadImageSource(file);
  try {
    const canvas = document.createElement("canvas");
    const g = canvas.getContext("2d", { willReadFrequently: true });
    if (!g) throw new Error("canvas 2D context unavailable");

    for (const [maxDim, opts] of [
      [2000, { ...BASE_OPTS, tryDenoise: true }],
      [1400, { ...BASE_OPTS, binarizer: "GlobalHistogram" as const, tryDenoise: true }],
      [900, BASE_OPTS],
    ] as [number, ReaderOptions][]) {
      const s = Math.min(1, maxDim / Math.max(loaded.width, loaded.height));
      const w = Math.max(1, Math.round(loaded.width * s));
      const h = Math.max(1, Math.round(loaded.height * s));
      canvas.width = w;
      canvas.height = h;
      g.drawImage(loaded.src, 0, 0, w, h);
      const hit = extractText(await readBarcodes(g.getImageData(0, 0, w, h), opts));
      if (hit) return hit;
    }

    // Centered square crop for full-screen phone screenshots with UI chrome around the QR.
    const side = Math.round(Math.min(loaded.width, loaded.height) * 0.78);
    const sx = Math.round((loaded.width - side) / 2);
    const sy = Math.round((loaded.height - side) / 2);
    const out = Math.min(1400, side);
    canvas.width = out;
    canvas.height = out;
    g.drawImage(loaded.src, sx, sy, side, side, 0, 0, out, out);
    const cropImg = g.getImageData(0, 0, out, out);
    const cropHit =
      extractText(await readBarcodes(cropImg, { ...BASE_OPTS, tryDenoise: true })) ??
      extractText(await readBarcodes(cropImg, { ...BASE_OPTS, binarizer: "GlobalHistogram", tryDenoise: true }));
    if (cropHit) return cropHit;
  } finally {
    loaded.release();
  }

  throw new Error("no QR code found in the image");
}

export type CameraScanner = {
  active: () => boolean;
  stop: () => void;
  flip: () => Promise<void>;
};

export type CameraScanOptions = {
  video: HTMLVideoElement;
  onStatus: (text: string) => void;
  onMultiCamera: (hasMultiple: boolean) => void;
  onDetect: (text: string) => void;
  onError: (message: string) => void;
  onStop: () => void;
};

async function openCameraStream(facing: Facing): Promise<MediaStream> {
  const attempts: MediaStreamConstraints[] = [
    { video: { facingMode: { ideal: facing }, width: { ideal: 1920, min: 640 }, height: { ideal: 1080, min: 480 } }, audio: false },
    { video: { facingMode: { ideal: facing } }, audio: false },
    { video: true, audio: false },
  ];
  let lastErr: unknown;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      lastErr = err;
      const name = (err as DOMException)?.name;
      if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "NotFoundError" || name === "DevicesNotFoundError") {
        throw err;
      }
    }
  }
  throw lastErr;
}

/** Starts a live camera viewfinder that auto-detects dense QR codes and closes itself on match. */
export async function startCameraQR(opts: CameraScanOptions, initialFacing: Facing = "environment"): Promise<CameraScanner> {
  let stream: MediaStream | null = null;
  let facing: Facing = initialFacing;
  let token = 0;
  let timer = 0;

  const stop = () => {
    token++;
    if (timer) { window.clearTimeout(timer); timer = 0; }
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    opts.video.srcObject = null;
    opts.onStop();
  };

  const launch = async (dir: Facing) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      opts.onError("Live camera needs HTTPS and camera access. Use Upload QR image instead.");
      stop();
      return;
    }
    const myToken = ++token;
    if (timer) { window.clearTimeout(timer); timer = 0; }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    opts.onStatus("Starting camera…");

    // Pre-load the WASM engine in parallel with the camera permission / startup.
    const zxingReady = getZXing();

    let s: MediaStream;
    try {
      s = await openCameraStream(dir);
    } catch (err) {
      stop();
      const name = (err as DOMException)?.name;
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        opts.onError("Camera permission denied. Use Upload QR image to pick a screenshot or photo instead.");
      } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        opts.onError("No camera found. Use Upload QR image to pick a screenshot or photo instead.");
      } else {
        opts.onError(`Could not start camera: ${(err as Error).message || name || "unknown error"}. Try Upload QR image.`);
      }
      return;
    }

    if (myToken !== token) {
      s.getTracks().forEach((t) => t.stop());
      return;
    }
    stream = s;
    facing = dir;

    // Request continuous autofocus where supported (improves close-up scanning of dense screen QRs).
    const track = s.getVideoTracks()[0];
    if (track?.applyConstraints) {
      track.applyConstraints({ advanced: [{ focusMode: "continuous" } as MediaTrackConstraintSet] }).catch(() => {});
    }

    opts.video.srcObject = s;
    try { await opts.video.play(); } catch { /* muted + playsinline handles mobile autoplay */ }

    navigator.mediaDevices.enumerateDevices?.().then((devs) => {
      if (myToken === token) opts.onMultiCamera(devs.filter((d) => d.kind === "videoinput").length >= 2);
    }).catch(() => {});

    let readBarcodes: ZXingReaderMod["readBarcodes"];
    try {
      ({ readBarcodes } = await zxingReady);
    } catch (err) {
      stop();
      opts.onError(`Could not load QR scanner: ${(err as Error).message}`);
      return;
    }
    if (myToken !== token) return;

    opts.onStatus("Point the camera at a task QR code — it loads automatically.");

    const canvas = document.createElement("canvas");
    const g = canvas.getContext("2d", { willReadFrequently: true });
    if (!g) { stop(); opts.onError("Canvas 2D unavailable."); return; }

    let step = 0;
    const tick = async () => {
      if (myToken !== token || !stream) return;
      try {
        const vw = opts.video.videoWidth, vh = opts.video.videoHeight;
        if (opts.video.readyState >= 2 && vw > 0 && vh > 0) {
          let hit: string | null = null;
          const mode = step++ % 3;
          if (mode === 0) {
            // Centered native-resolution crop inside the viewfinder square (sharpest for dense QRs).
            const side = Math.round(Math.min(vw, vh) * 0.72);
            const sx = Math.round((vw - side) / 2);
            const sy = Math.round((vh - side) / 2);
            const out = Math.min(1080, side);
            canvas.width = out;
            canvas.height = out;
            g.drawImage(opts.video, sx, sy, side, side, 0, 0, out, out);
            hit = extractText(await readBarcodes(g.getImageData(0, 0, out, out), BASE_OPTS));
          } else if (mode === 1) {
            // Full frame (catches QRs held closer or off-center).
            const sc = Math.min(1, 1280 / Math.max(vw, vh));
            const w = Math.max(1, Math.round(vw * sc));
            const h = Math.max(1, Math.round(vh * sc));
            canvas.width = w;
            canvas.height = h;
            g.drawImage(opts.video, 0, 0, w, h);
            hit = extractText(await readBarcodes(g.getImageData(0, 0, w, h), BASE_OPTS));
          } else {
            // Centered crop with GlobalHistogram + denoise (helps against screen glare / moire).
            const side = Math.round(Math.min(vw, vh) * 0.72);
            const sx = Math.round((vw - side) / 2);
            const sy = Math.round((vh - side) / 2);
            const out = Math.min(960, side);
            canvas.width = out;
            canvas.height = out;
            g.drawImage(opts.video, sx, sy, side, side, 0, 0, out, out);
            hit = extractText(
              await readBarcodes(g.getImageData(0, 0, out, out), {
                ...BASE_OPTS,
                binarizer: "GlobalHistogram",
                tryDenoise: true,
              }),
            );
          }
          if (hit && myToken === token) {
            stop();
            opts.onDetect(hit);
            return;
          }
        }
      } catch { /* ignore transient frame error and keep scanning */ }
      if (myToken === token && stream) {
        timer = window.setTimeout(tick, 120);
      }
    };
    timer = window.setTimeout(tick, 150);
  };

  await launch(facing);

  return {
    active: () => stream !== null,
    stop,
    flip: async () => {
      await launch(facing === "environment" ? "user" : "environment");
    },
  };
}
