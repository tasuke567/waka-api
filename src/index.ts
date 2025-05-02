// Express + Weka API (STRING-friendly ✅)
import crypto from "node:crypto";
import express from "express";
import multer from "multer";
import { execFile, execSync } from "node:child_process";
import fs, {
  promises as fsp,
  existsSync,
  mkdirSync,
  statSync,
  readdirSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "url";
import csvParser from "csv-parser";
import { v4 as uuidv4 } from "uuid";
import cors from "cors";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ?? 3000;

const MODEL = path.join(__dirname, "../model/myJ48.model");
const HEADER_PATH = path.join(__dirname, "../model/header.arff");
const WEKA_JAR = path.join(__dirname, "../model/weka.jar");
const MTJ_JAR = path.join(__dirname, "../model/mtj-1.0.4.jar");
const WEKA_CP = [WEKA_JAR, MTJ_JAR].join(path.delimiter);
const CLASS_ATTR = "Current_brand";

const UPLOAD_DIR = path.join(process.cwd(), "uploads");
const trainDir = path.join(UPLOAD_DIR, "train");
const javaPath = process.env.JAVA_CMD ?? "java";

// ──────────────────────────────────────────────────────────────────────────
// bootstrap dirs / java
[UPLOAD_DIR, trainDir].forEach((d) => {
  if (!existsSync(d)) mkdirSync(d, { recursive: true });
});

function checkJava() {
  try {
    execSync("java -version");
    console.log("Java found ✅");
  } catch (e) {
    throw new Error(
      `Java check failed: ${e instanceof Error ? e.message : String(e)}`
    );
  }
}

// ──────────────────────────────────────────────────────────────────────────
// multer
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOAD_DIR),
  filename: (_, file, cb) =>
    cb(null, `${crypto.randomUUID()}${path.extname(file.originalname)}`),
});
const MAX_MB = Number(process.env.MAX_UPLOAD_MB ?? 50); // default 50 MB

const upload = multer({
  storage,
  limits: { fileSize: MAX_MB * 1024 * 1024 },
  fileFilter: (_, file, cb) => {
    const allowed = [
      "text/csv",
      "text/plain",
      "application/octet-stream",
      "application/x-arff",
      "application/zip", // ถ้าจะให้รับ ZIP ด้วย
    ];
    cb(null, allowed.includes(file.mimetype));
  },
});

// ──────────────────────────────────────────────────────────────────────────
// helpers
const esc = (v: string): string => {
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

// build ARFF (train / predict) — keeps STRING everywhere
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
    ? generateHeader(rows, cols) // new header
    : fs.readFileSync(HEADER_PATH, "utf8"); // use stored header (STRING)

  // attribute lookup for predict validity
  const attrMap: Record<string, Set<string>> = {};
  if (!isTrain) {
    headerText
      .split("\n")
      .filter((l) => l.startsWith("@ATTRIBUTE"))
      .forEach((l) => {
        const name = l.split(/\s+/)[1];
        const match = l.match(/\{(.+)\}/);
        if (match)
          attrMap[name] = new Set(match[1].split(",").map((s) => s.trim()));
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
      const line = cols
        .map((c) => {
          const raw = c === CLASS_ATTR && !isTrain ? "?" : r[c] ?? "?";
          const val = esc(raw);
          return !isTrain && attrMap[c] && !attrMap[c].has(val) ? "?" : val;
        })
        .join(",");
      ws.write(line + "\n");
    });
    ws.end();
  });
  return out;
}

// header generator (forces some cols → STRING)
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
      const vals = [...new Set(rows.map((r) => esc(r[col] ?? "?")))].sort();
      lines.push(`@ATTRIBUTE ${col} {${vals.join(",")}}`);
    }
  }
  lines.push("", "@DATA");
  return lines.join("\n");
}

// run prediction
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
    "weka.classifiers.evaluation.output.prediction.CSV -decimals 6 -distribution",
  ];

  return new Promise((ok, err) => {
    execFile(javaPath, args, { encoding: "utf8" }, (e, stdout, stderr) => {
      if (e || /Exception|Error/i.test(stderr)) {
        return err(new Error(`Weka failed:\n${stderr}\n${stdout}`));
      }

      const lines = stdout.trim().split("\n").filter(Boolean);
      const idx = lines.findIndex((l) => l.startsWith("inst#"));
      if (idx === -1 || idx + 1 >= lines.length)
        return err(new Error("No prediction rows"));

      const header = lines[idx].split(",").map((s) => s.trim().toLowerCase());
      const data = lines[idx + 1].split(",");

      const predIdx = header.findIndex((h) => h === "predicted");
      if (predIdx === -1) return err(new Error("Missing 'predicted' column"));

      // Weka เก็บรูปแบบ "12:Xiaomi"
      const rawPred = data[predIdx] ?? "";
      const label = rawPred.includes(":")
        ? rawPred.split(":").slice(1).join(":").trim() // ทุกอย่างหลัง ':' = ชื่อแบรนด์
        : rawPred.trim();
      if (!label) return err(new Error("Prediction label missing"));

      const dist: Record<string, number> = {};
      header.forEach((h, i) => {
        if (h.startsWith("prob_")) {
          const k = h.replace("prob_", "").trim();
          const v = parseFloat(data[i]);
          dist[k] = isNaN(v) ? 0 : v;
        }
      });
      ok({ label, distribution: dist });
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────
// express routes
const app = express();
app.use(cors()); // 💥 ต้องอยู่บนสุด ก่อน route ใด ๆ

// หรือตั้งให้ปลอดภัยขึ้นแบบเฉพาะ origin
app.use(
  cors({
    origin: "http://localhost:5173",
    methods: ["GET", "POST"],
  })
);

app.post("/predict", upload.single("file"), async (req, res) => {
  try {
    const arff = await buildArff(req.file!.path, false);
    const fname = `predict-${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}-${uuidv4()}.arff`;
    const final = path.join(UPLOAD_DIR, fname);
    fs.copyFileSync(arff, final);

    const result = await wekaPredict(final, MODEL);
    res.json({ prediction: result });
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: "Weka error", message: String(e) });
  }
});

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
      "weka.classifiers.trees.J48",
      "-t",
      final,
      "-d",
      MODEL,
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

    // 2) สั่ง Weka ให้พ่น CSV Prediction “ทุกอินสแตนซ์”
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

// ──────────────────────────────────────────────────────────────────────────
checkJava();
app.listen(PORT, () => console.log(`🚀  http://localhost:${PORT}`));
// ──────────────────────────────────────────────────────────────────────────
