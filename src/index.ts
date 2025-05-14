// Express + Weka API (STRING‑friendly ✅)
import crypto from "node:crypto";
import express from "express";
import type { RequestHandler } from "express";
import { Parser } from "json2csv";
import multer from "multer";
import { execFile, execSync } from "node:child_process";
import fs, { existsSync, mkdirSync, statSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "url";
import csvParser from "csv-parser";
import { v4 as uuidv4 } from "uuid";
import cors from "cors";
import type { ParsedQs } from "qs";
import { Prisma } from "@prisma/client";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ?? 3000;

// ────────── paths / consts ──────────
const MODEL = path.join(__dirname, "../model/myJ48.model");
const HEADER_PATH = path.join(__dirname, "../model/header.arff");
const WEKA_JAR = path.join(__dirname, "../model/weka.jar");
const MTJ_JAR = path.join(__dirname, "../model/mtj-1.0.4.jar");
const WEKA_CP = [WEKA_JAR, MTJ_JAR].join(path.delimiter);
const CLASS_ATTR = "Current_brand";

const UPLOAD_DIR = path.join(process.cwd(), "uploads");
const trainDir = path.join(UPLOAD_DIR, "train");
const feedbackDir = path.join(UPLOAD_DIR, "feedback");
const javaPath = process.env.JAVA_CMD ?? "java";

import { authRouter } from "./routes/authRoutes.js";
import { verifyToken } from "./middleware/verifyToken.js";
import { prisma } from "./db.js";
import "dotenv/config";

// ────────── bootstrap ──────────
[UPLOAD_DIR, trainDir, feedbackDir].forEach((d) => {
  if (!existsSync(d)) mkdirSync(d, { recursive: true });
});

function checkJava() {
  try {
    execSync("java -version");
    console.log("Java found ✅");
  } catch (e) {
    throw new Error(`Java check failed: ${e}`);
  }
}

// ────────── multer ──────────
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOAD_DIR),
  filename: (_, f, cb) =>
    cb(null, `${crypto.randomUUID()}${path.extname(f.originalname)}`),
});
const upload = multer({
  storage,
  limits: { fileSize: Number(process.env.MAX_UPLOAD_MB ?? 50) * 1024 * 1024 },
  fileFilter: (_, file, cb) =>
    cb(
      null,
      [
        "text/csv",
        "text/plain",
        "application/octet-stream",
        "application/x-arff",
        "application/zip",
      ].includes(file.mimetype)
    ),
});

// ────────── helpers ──────────
const esc = (v: string) => {
  let s = (v ?? "?")
    .trim()
    .replace(/\u00A0/g, " ")
    .replace(/,+/g, "")
    .replace(/\s+/g, " ");
  if (!s || s === "?") return "?";
  s = s.replace(/'/g, "\\'");
  return /[\s,{}]/.test(s) ? `'${s}'` : s;
};
const parseArffHeader = (txt: string) =>
  txt
    .split("\n")
    .filter((l) => l.trim().startsWith("@ATTRIBUTE"))
    .map((l) => l.trim().split(/\s+/)[1]);
const clean = (s: string) => s.replace(/^'+|'+$/g, "").trim();

// ────────── CLASS_VALUES once ──────────
const CLASS_VALUES: string[] = (() => {
  const txt = fs.readFileSync(HEADER_PATH, "utf8");
  const line = txt
    .split("\n")
    .find((l) => l.startsWith(`@ATTRIBUTE ${CLASS_ATTR} `));
  const m = line?.match(/\{(.+)\}/);
  return m ? m[1].split(",").map((s) => s.trim()) : [];
})();

// ────────── buildArff ──────────
async function buildArff(csvPath: string, isTrain: boolean): Promise<string> {
  const rows: Record<string, string>[] = [];
  await new Promise<void>((ok, err) => {
    fs.createReadStream(csvPath)
      .pipe(
        csvParser({
          mapHeaders: ({ header }) =>
            header
              .replace(/^\uFEFF/, "")
              .trim()
              .replace(/\s+/g, "_"),
          mapValues: ({ value }) =>
            value
              .trim()
              .replace(/,/g, "")
              .replace(/\u00A0/g, " ")
              .replace(/\s+/g, " ")
              .normalize("NFKC"),
        })
      )
      .on("data", (r) => rows.push(r))
      .on("end", ok)
      .on("error", err);
  });

  const cols = isTrain
    ? Object.keys(rows[0])
        .filter((k) => k !== CLASS_ATTR)
        .concat(CLASS_ATTR)
    : parseArffHeader(fs.readFileSync(HEADER_PATH, "utf8"));
  const headerText = isTrain
    ? generateHeader(rows, cols)
    : fs.readFileSync(HEADER_PATH, "utf8");

  const attrMap: Record<string, Set<string>> = {} as Record<
    string,
    Set<string>
  >;
  if (!isTrain) {
    headerText
      .split("\n")
      .filter((l) => l.startsWith("@ATTRIBUTE"))
      .forEach((l) => {
        const name = l.split(/\s+/)[1];
        const m = l.match(/\{(.+)\}/);
        if (m) attrMap[name] = new Set(m[1].split(",").map((s) => s.trim()));
      });
  }

  const out = path.join(UPLOAD_DIR, `${uuidv4()}.arff`);
  await new Promise<void>((ok, err) => {
    const ws = fs.createWriteStream(out);
    ws.on("error", err).on("finish", ok);
    ws.write(headerText.trim());
    if (!headerText.toLowerCase().includes("@data")) ws.write("\n@DATA");
    ws.write("\n");
    rows.forEach((r) => {
      ws.write(
        cols
          .map((c) => {
            const raw = c === CLASS_ATTR && !isTrain ? "?" : r[c] ?? "?";
            const val = esc(raw);
            return !isTrain && attrMap[c] && !attrMap[c].has(val) ? "?" : val;
          })
          .join(",") + "\n"
      );
    });
    ws.end();
  });
  return out;
}

function generateHeader(
  rows: Record<string, string>[],
  cols: string[],
  forceString = new Set(["Top3_smartphone_activities", "Frequent_apps"])
) {
  const lines = ["@RELATION smartphone"];
  for (const col of cols) {
    if (forceString.has(col)) {
      lines.push(`@ATTRIBUTE ${col} STRING`);
    } else {
      const uniq = Array.from(new Set(rows.map((r) => esc(r[col] ?? "?"))));
      const vals = uniq.filter((v: string) => v !== "?").sort();
      lines.push(`@ATTRIBUTE ${col} {${vals.join(",")}}`);
    }
  }
  lines.push("", "@DATA");
  return lines.join("\n");
}

// ────────── wekaPredict (keep 6‑decimals) ──────────
function wekaPredict(
  arff: string,
  model: string
): Promise<{ label: string; distribution: Record<string, number> }> {
  const args = [
    "-Xmx1G",
    "-cp",
    WEKA_CP.replace(/\\/g, "/"),
    "weka.classifiers.meta.FilteredClassifier",
    "-l",
    model.replace(/\\/g, "/"),
    "-T",
    arff.replace(/\\/g, "/"),
    "-c",
    "last",
    "-classifications",
    "weka.classifiers.evaluation.output.prediction.CSV -decimals 8 -distribution",
  ];
  return new Promise((ok, err) => {
    execFile(javaPath, args, { encoding: "utf8" }, (e, stdout, stderr) => {
      if (e || /Exception|Error/i.test(stderr))
        return err(new Error(`Weka failed:\n${stderr}\n${stdout}`));
      const lines = stdout.trim().split("\n").filter(Boolean);
      const idx = lines.findIndex((l) => l.startsWith("inst#"));
      if (idx === -1 || idx + 1 >= lines.length)
        return err(new Error("No prediction rows"));

      const header = lines[idx].split(",").map((s) => s.trim().toLowerCase());
      const data = lines[idx + 1].split(",");

      // ── predicted label ──
      const predIdx = header.findIndex((h) => h === "predicted");
      if (predIdx === -1) return err(new Error("Missing 'predicted' column"));
      const rawPred = data[predIdx] ?? "";
      const label = rawPred.includes(":")
        ? rawPred.split(":").slice(1).join(":").trim()
        : rawPred.trim();
      if (!label) return err(new Error("Prediction label missing"));

      // ── probability distribution ──
      const dist: Record<string, number> = {};

      // A) columns starting with prob_ / prob:
      header.forEach((h, i) => {
        const m = h.match(/^prob[:_(]?(.+?)[)_]?$/i);
        if (m) {
          const v = parseFloat(data[i]);
          dist[clean(m[1])] = Number.isNaN(v) ? 0 : Number(v.toFixed(6));
        }
      });

      // B) following the "distribution" marker
      const dIdx = header.findIndex((h) => h === "distribution");
      if (dIdx !== -1 && CLASS_VALUES.length) {
        const start = dIdx ;
        for (let i = 0; i < CLASS_VALUES.length; i++) {
          const num = parseFloat((data[start + i] || "").replace("*", ""));
          if (!isNaN(num))
            dist[clean(CLASS_VALUES[i])] = Number(num.toFixed(6));
        }
      }
      console.log(header, data);
      console.log({ header, data, CLASS_VALUES });

      ok({ label, distribution: dist });
    });
  });
}

// ────── Role guard (วางไว้เหนือ routes) ──────
const adminOnly: RequestHandler = (req, res, next) => {
  if (!req.user || req.user.role !== "ADMIN") {
    res.status(403).json({ error: "Admin only 🔒" });
    return;
  }
  next();
};

/* ---------- PREDICT HANDLER ---------- */
const predictHandler: RequestHandler = async (req, res) => {
  try {
    /* 1) สร้าง ARFF + ทำนาย */
    const arff = await buildArff(req.file!.path, false);
    const fname =
      "predict-" +
      new Date().toISOString().replace(/[:.]/g, "-") +
      "-" +
      uuidv4() +
      ".arff";
    const final = path.join(UPLOAD_DIR, fname);
    fs.copyFileSync(arff, final);

    const result = await wekaPredict(final, MODEL);

    /* 2) เตรียม payload สำหรับ Prisma */
    const data: Prisma.QuestionnaireUncheckedCreateInput = {
      rawCsvPath: final,
      userId: req.user?.id ?? null, // ถ้าไม่มี token ⇒ null
      prediction: {
        create: {
          label: result.label,
          distribution: result.distribution as any,
        },
      },
    };

    /* 3) บันทึกลง DB */
    const q = await prisma.questionnaire.create({ data });

    res.json({ questionnaireId: q.id, prediction: result });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Weka error", message: String(e) });
  }
};

// ──────────────────────────────────────────────────────────────────────────
// express routes

const app = express();
const ALLOWED = [
  "http://localhost:5173",
  "https://brand-predictor.netlify.app",
];

app.use(
  cors({
    origin: (origin, callback) => {
      // ถ้าไม่มี origin (เช่น postman) หรือ origin อยู่ใน whitelist → อนุญาต
      if (!origin || ALLOWED.includes(origin)) {
        return callback(null, true);
      }
      callback(new Error("Not allowed by CORS"));
    },
    credentials: true, // สำหรับ auth header / cookie
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  })
);
// ก่อนประกาศทุก route
app.use(express.json()); // ✔️ parse application/json
app.use(
  express.urlencoded({
    // (เผื่อ form-urlencoded ธรรมดา)
    extended: true,
  })
);

app.post(
  "/predict",
  upload.single("file"), // multipart/form-data field = file
  predictHandler // <- ส่ง handler ที่เราเตรียมไว้
);
app.post("/train", upload.single("file"), async (req, res) => {
  try {
    const arff = await buildArff(req.file!.path, true);
    const fname = `train-${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}-${uuidv4()}.arff`;
    const final = path.join(trainDir, fname);
    fs.copyFileSync(arff, final);

    // update header for future predictions
    const header = fs.readFileSync(arff, "utf8").split("@DATA")[0];
    fs.writeFileSync(HEADER_PATH, header);

    const args = [
      "-Xmx1G",
      "-cp",
      WEKA_CP.replace(/\\/g, "/"),
      "weka.classifiers.meta.FilteredClassifier",

      "-F",
      "weka.filters.unsupervised.attribute.StringToNominal -R first-last",

      "-W",
      "weka.classifiers.trees.RandomForest",

      "-t",
      final, // <-- training file
      "-d",
      MODEL, // บันทึกโมเดล
      "-c",
      "last",
      "-x",
      "10",
      "-o",
    ];

    await new Promise<void>((ok, err) => {
      execFile(javaPath, args, { encoding: "utf8" }, (e, stdout, stderr) => {
        if (e || /Exception/i.test(stderr)) return err(new Error(stderr));
        console.log(stdout);
        ok();
      });
    });

    if (!existsSync(MODEL)) throw new Error(`Model not saved: ${MODEL}`);
    res.json({ saved: final, model: MODEL });
  } catch (e: any) {
    res.status(500).json({ error: String(e) });
  }
});

// util endpoints
app.get("/predict-history", (_, res) => {
  res.json(
    readdirSync(UPLOAD_DIR)
      .filter((f) => f.startsWith("predict-") && f.endsWith(".arff"))
      .map((f) => ({
        file: f,
        time: statSync(path.join(UPLOAD_DIR, f)).birthtime,
      }))
  );
});

app.get("/train-history", (_, res) => {
  res.json(
    readdirSync(trainDir)
      .filter((f) => f.startsWith("train-") && f.endsWith(".arff"))
      .map((f) => ({
        file: f,
        time: statSync(path.join(trainDir, f)).birthtime,
      }))
  );
});

app.get("/model-info", (_, res) => {
  const header = fs.readFileSync(HEADER_PATH, "utf8");
  const cols = parseArffHeader(header);
  const cls = cols.at(-1)!;
  const vals =
    header
      .split("\n")
      .find((l) => l.startsWith(`@ATTRIBUTE ${cls} `))
      ?.match(/\{(.*?)\}/)?.[1]
      ?.split(",") ?? [];
  res.json({ classAttr: cls, values: vals });
});

/* ------------------------------------------------------------------
    🆕  /predict-batch   (POST multipart/form-data, field = file)
    — รับไฟล์ test CSV/ARFF หลายแถว → คืน predictions[] + (option) dist
-------------------------------------------------------------------*/
app.post("/predict-batch", upload.single("file"), async (req, res) => {
  try {
    // 1) สร้าง/ใช้ ARFF เหมือนเดิม แต่จะอ่านทุกแถว
    const arffPath = await buildArff(req.file!.path, false);

    // 2) สั่ง Weka ให้พ่น CSV Prediction "ทุกอินสแตนซ์"
    const args = [
      "-Xmx2G",
      "-cp",
      WEKA_CP.replace(/\\/g, "/"),
      "weka.classifiers.meta.FilteredClassifier",
      "-l",
      MODEL.replace(/\\/g, "/"),
      "-T",
      arffPath.replace(/\\/g, "/"),
      "-c",
      "last",
      "-classifications",
      // -p 0 = output ทุกแถว  ;  -distribution = แจก probs
      '"weka.classifiers.evaluation.output.prediction.CSV -decimals 6 -distribution"',
      "-p",
      "0",
    ];

    execFile(
      javaPath,
      args,
      { encoding: "utf8", shell: true },
      (err, stdout, stderr) => {
        if (err || /Exception|Error/i.test(stderr)) {
          return res.status(500).json({ error: stderr, stdout });
        }
        /* ── Parse ── */
        const lines = stdout
          .trim()
          .split("\n")
          .filter((l) => l.startsWith("inst#") || /^\d/.test(l));
        const header = lines[0].split(",").map((s) => s.trim());
        const idxPred = header.findIndex(
          (h) => h.toLowerCase() === "predicted"
        );
        const probIdx = header
          .map((h, i) =>
            h.startsWith("prob_") ? [h.replace("prob_", ""), i] : null
          )
          .filter(Boolean) as [string, number][];

        const preds = lines.slice(1).map((l) => {
          const cols = l.split(",");
          const raw = cols[idxPred] ?? "";
          const label = raw.includes(":")
            ? raw.split(":").slice(1).join(":").trim()
            : raw.trim();
          const dist: Record<string, number> = {};
          probIdx.forEach(([k, i]) => (dist[k] = parseFloat(cols[i] ?? "0")));
          return { label, distribution: dist };
        });

        res.json({ total: preds.length, predictions: preds });
      }
    );
  } catch (e: any) {
    res.status(500).json({ error: String(e) });
  }
});

/* ---------- ชนิด ---------- */
interface FeedbackBody {
  prediction: string;
  uiEase: number;
  satisfaction: number;
  clarity: number;
}

interface FeedbackOk {
  ok: true;
  id: string;
}
interface FeedbackErr {
  error: string;
}
type FeedbackRes = FeedbackOk | FeedbackErr;

/* ---------- handler ---------- */
const feedbackHandler: RequestHandler<
  {},
  FeedbackRes,
  FeedbackBody,
  ParsedQs
> = async (req, res) => {
  const { prediction, uiEase, satisfaction, clarity } = req.body;

  if (
    typeof prediction !== "string" ||
    [uiEase, satisfaction, clarity].some((v) => typeof v !== "number")
  ) {
    res.status(400).json({ error: "Invalid payload" });
    return;
  }

  const q = await prisma.predictionResult.findFirst({
    where: { label: prediction },
    orderBy: { createdAt: "desc" },
  });
  if (!q) {
    res.status(404).json({ error: "No prediction found" });
    return;
  }

  const fb = await prisma.feedback.create({
    data: {
      questionnaireId: q.questionnaireId,
      uiEase,
      satisfaction,
      clarity,
    },
  });

  res.json({ ok: true, id: fb.id.toString() });
};

/* ---------- route ---------- */
app.post("/feedback", feedbackHandler);

// simple GET feedback list (dev only)
app.get("/feedback", (_, res) => {
  const files = readdirSync(feedbackDir);
  const list = files.map((f) =>
    JSON.parse(fs.readFileSync(path.join(feedbackDir, f), "utf8"))
  );
  res.json(list);
});

// === Public ===
app.use("/auth", authRouter);
app.get("/stats/brands", async (_, res) => {
  const agg = await prisma.predictionResult.groupBy({
    by: ["label"],
    _count: { _all: true },
  });

  const sorted = agg
    .sort((a, b) => (b._count._all ?? 0) - (a._count._all ?? 0))
    .map((a) => ({ brand: a.label, total: a._count._all }));

  res.json(sorted);
});

// === Protected example ===
app.get("/profile", verifyToken, async (req, res) => {
  const { id } = req.user!; // TS now knows it exists
  const user = await prisma.user.findUnique({ where: { id } });
  res.json(user);
});

// กลุ่ม /admin/*
const admin = express.Router();
app.use("/admin", verifyToken, adminOnly, admin);

// --- 1) list questionnaire + prediction ---
admin.get("/questionnaire", async (_, res) => {
  const list = await prisma.questionnaire.findMany({
    include: { prediction: true, feedbacks: true, user: true },
    orderBy: { createdAt: "desc" },
  });
  res.json(list);
});

// --- 2) delete questionnaire ---
admin.delete("/questionnaire/:id", async (req, res) => {
  const id = Number(req.params.id);
  await prisma.questionnaire.delete({ where: { id } });
  res.json({ ok: true });
});

// --- 3) export CSV report ---
// npm i json2csv
admin.get("/report/export", async (_, res) => {
  const data = await prisma.predictionResult.findMany({
    include: { questionnaire: { include: { user: true } } },
  });
  const parser = new Parser();
  const csv = parser.parse(
    data.map((d) => ({
      qId: d.questionnaireId,
      brand: d.label,
      createdAt: d.createdAt,
      ...(d.distribution as Record<string, number>),
    }))
  );
  res.setHeader("Content-Type", "text/csv");
  res.setHeader("Content-Disposition", `attachment; filename=report.csv`);
  res.send(csv);
});

// --- 4) hot-swap model ---
admin.post("/model", upload.single("file"), (req, res) => {
  try {
    fs.copyFileSync(req.file!.path, MODEL);
    res.json({ ok: true, note: "Model replaced ✅" });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});
admin.delete("/model", (_, res) => {
  if (existsSync(MODEL)) fs.unlinkSync(MODEL);
  res.json({ ok: true, note: "Model deleted" });
});

// ──────────────────────────────────────────────────────────────────────────
checkJava();
app.listen(PORT, () => console.log(`🚀  http://localhost:${PORT}`));
// ──────────────────────────────────────────────────────────────────────────
