/* ----------------------------------------------------------
   Express + Weka API  ─  /predict  (POST multipart/form-data)
   ---------------------------------------------------------- */
import crypto from "node:crypto";
import express from "express";
import multer from "multer";
import { execFile } from "node:child_process";
import fs, { promises as f } from "node:fs";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "url";
import csvParser from "csv-parser";
import { execSync } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ?? 3000;
const WEKA_JAR =
  process.env.WEKA_JAR || path.join(__dirname, "../model/weka.jar");
const MTJ_JAR = process.env.MTJ_JAR ?? "model/mtj-1.0.4.jar";
const MODEL = path.join(__dirname, "../model/myJ48.model");
const HEADER = fs.readFileSync(
  path.join(__dirname, "../model/header.arff.tpl"),
  "utf8"
);
const CLASS_ATTR = "Current_brand"; // ชื่อคอลัมน์ที่เป็น class label
const WEKA_CP = [WEKA_JAR, MTJ_JAR].join(path.delimiter);

interface Prediction {
  label: string; // เช่น "Apple"
  distribution: number[]; // เช่น [0.10, 0.05, 0.40, 0.15, 0.30]
}

function checkJava() {
  try {
    execSync("java -version", { stdio: "inherit" });
  } catch (e) {
    throw new Error(
      "Java Runtime Environment (JRE) is required but not found.\n" +
        "Install with: sudo apt-get install openjdk-17-jre"
    );
  }
}
/* ---------- multer ---------- */
// Ensure uploads directory exists
const uploadDir = path.join(__dirname, "uploads");
if (!existsSync(uploadDir)) {
  mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // Double-check directory exists on each upload
    if (!existsSync(uploadDir)) {
      mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const name = crypto.randomUUID() + ext;
    cb(null, name);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 5_000_000 },
  fileFilter: (_, file, cb) => {
    const allowedMimes = [
      "text/csv",
      "text/plain",
      "application/octet-stream",
      "application/x-arff",
    ];

    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type: ${file.mimetype}`));
    }
  },
});

/* ---------- helpers ---------- */
async function buildArff(csvPath: string, isTrain: boolean): Promise<string> {
  // Use absolute paths consistently
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const uploadDir = path.join(__dirname, "uploads");

  // 0) Verify input file exists
  if (!existsSync(csvPath)) {
    throw new Error(`Input file not found: ${csvPath}`);
  }

  // 1) Load CSV rows
  const rows: Record<string, string>[] = [];
  await new Promise<void>((res, rej) => {
    fs.createReadStream(csvPath, { encoding: "utf8" })
      .pipe(
        csvParser({
          mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim(),
        })
      )
      .on("headers", (hdrs: string[]) => {
        console.log("🔍 CSV headers:", hdrs);
      })
      .on("data", (row) => rows.push(row))
      .on("end", () => res())
      .on("error", rej);
  });

  console.log(`✨ Parsed ${rows.length} row(s) from CSV`);
  if (!rows.length) throw new Error("No data in CSV");

  // 2) Load header template
  const headerPath = path.join(__dirname, "../model/header.arff.tpl");
  const HEADER = fs.readFileSync(headerPath, "utf8");

  // 3) Extract attributes
  const cols = HEADER.split("\n")
    .filter((l) => l.trim().startsWith("@ATTRIBUTE"))
    .map((l) => l.trim().split(/\s+/)[1]);

  // 4) Prepare ARFF path
  const arffPath = path.join(uploadDir, `${crypto.randomUUID()}.arff`);

  // Ensure upload directory exists
  if (!existsSync(uploadDir)) {
    mkdirSync(uploadDir, { recursive: true });
  }

  // 5) Write file with proper error handling
  const ws = fs.createWriteStream(arffPath, { encoding: "utf8" });

  return new Promise<string>((resolve, reject) => {
    ws.on("error", reject).on("finish", () => {
      console.log("✅ ARFF generated at", arffPath);
      resolve(path.resolve(arffPath));
    });

    // Write header and data
    ws.write(HEADER.trim() + "\n@DATA\n");

    // Process rows
    for (const r of rows) {
      const line = cols
        .map((col) => {
          if (col === CLASS_ATTR) {
            if (isTrain) {
              const v = r[col];
              if (!v) throw new Error(`Missing class ${col}`);
              return /[\s,{}]/.test(v) ? `'${v}'` : v;
            }
            return "?";
          }
          const v = r[col]!;
          return /[\s,{}]/.test(v) ? `'${v}'` : v;
        })
        .join(",");

      ws.write(line + "\n");
    }

    ws.end();
  });
}

function wekaPredict(arff: string, modelPath: string): Promise<Prediction> {
  return new Promise((resolve, reject) => {
    const javaPath = "/usr/bin/java";
    const args = [
      "-Xmx1G",
      "-cp",
      WEKA_CP,
      "weka.classifiers.trees.J48",
      "-l",
      modelPath, // โหลดโมเดล
      "-T",
      arff, // test arff
      "-p",
      "0", // print คอลัมน์ last (class)
      "-distribution", // (เลือก) ให้โชว์เปอร์เซ็นต์
    ];

    execFile(javaPath, args, { encoding: "utf8" }, (err, stdout, stderr) => {
      if (err) {
        return reject(new Error(stderr || err.message));
      }
      console.log("=== WEKA OUTPUT ===\n", stdout);

      // 1) ลองแมทช์แบบมี distribution
      // แมทช์ distribution
      let m = stdout.match(/^\s*\d+\s+\S+\s+(\S+)\s+\S+\s+\[([^\]]+)\]/m);
      if (m) {
        const label = m[1].split(":").pop()!;
        const dist = m[2].split(",").map(parseFloat);
        return resolve({ label, distribution: dist });
      }
      // ถ้าไม่มี distribution ก็ fallback แมทช์แค่ label
      m = stdout.match(/^\s*\d+\s+\S+\s+(\S+)/m);

      if (m) {
        const raw = m[1].includes(":") ? m[1].split(":")[1] : m[1];
        return resolve({ label: raw, distribution: [] });
      }

      reject(new Error("No valid prediction found in Weka output"));
    });
  });
}

function wekaTrain(
  algorithm: string,
  arffPath: string,
  modelName: string,
  options: string[] = []
) {
  return new Promise<string>((ok, bad) => {
    // สร้างโฟลเดอร์เก็บโมเดลถ้ายังไม่มี
    const modelDir = path.resolve("model");
    if (!existsSync(modelDir)) fs.mkdirSync(modelDir, { recursive: true });

    const outModel = path.join(modelDir, modelName);
    const args = [
      "-Xmx1G",
      "-cp",
      WEKA_CP,
      algorithm, // e.g. "weka.classifiers.bayes.NaiveBayes"
      "-t",
      arffPath,
      "-d",
      outModel, // บันทึกไฟล์ .model
      ...options, // tuning options ถ้ามี
    ];

    console.log("Training CMD:", ["java", ...args].join(" "));
    execFile("java", args, { encoding: "utf8" }, (e, out, err) => {
      if (err) console.error("TRAIN-ERR\n", err);
      if (e) return bad(err || e.message);
      console.log("TRAIN-OUT\n", out);
      ok(outModel); // ส่ง path ของโมเดลกลับ
    });
  });
}

/* ---------- check model ---------- */
if (!existsSync(MODEL)) throw new Error("Model not found: " + MODEL);

/* ---------- route ------------ */
const app = express();

app.post("/predict", upload.single("file"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "file missing" });
    return;
  }

  const tmp = await buildArff(req.file.path, false); // isTrain = false

  try {
    const brand = await wekaPredict(tmp, MODEL);
    res.json({ brand });
  } catch (e) {
    const errorMessage = `Prediction failed: ${String(e)}`;
    console.error(errorMessage);
    res.status(500).json({
      error: errorMessage,
      details: e instanceof Error ? e.stack : undefined,
    });
  } finally {
    if (req.file?.path) {
      await f.unlink(req.file.path).catch(console.error);
    }
    if (tmp) {
      await f.unlink(tmp).catch(console.error);
    }
  }
});

app.post("/train", upload.single("file"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "file missing" });
    return;
  }

  const { algorithm = "weka.classifiers.trees.J48", modelName } = req.body;
  const options = req.body.options
    ? (req.body.options as string).trim().split(/\s+/)
    : [];

  try {
    // prepare ARFF
    const ext = path.extname(req.file.originalname).toLowerCase();
    const arffPath =
      ext === ".arff"
        ? path.resolve(req.file.path).replace(/\\/g, "/")
        : await buildArff(req.file.path, true); // isTrain = true

    // train
    const modelFile = modelName ?? crypto.randomUUID() + ".model";
    const modelPath = await wekaTrain(algorithm, arffPath, modelFile, options);

    res.json({ model: modelPath });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  } finally {
    // cleanup upload and temp ARFF
    await f.rm(req.file.path, { force: true });
    // if buildArff created a different path, remove it too
    // await f.rm(tmp, { force: true }); // not needed if we return the path from buildArff
    // (you could track and remove inside buildArff instead)
  }
});

app.get("/model-info", (req, res) => {
  const tempArff = path.join("uploads", "empty.arff");
  // เขียน ARFF เปล่าจาก HEADER template
  const header = HEADER.includes("@data") ? HEADER : HEADER + "\n@DATA\n";
  fs.writeFileSync(tempArff, header, "utf8");
  const args = [
    "-cp",
    WEKA_CP,
    "weka.classifiers.trees.J48",
    "-l",
    MODEL,
    "-T",
    path.resolve(tempArff).replace(/\\/g, "/"),
  ];
  execFile(
    "java",
    ["-Xmx1G", ...args],
    { encoding: "utf8" },
    (err, stdout, stderr) => {
      const outText = [stdout?.trim(), stderr?.trim(), err ? err.message : null]
        .filter(Boolean)
        .join("\n\n---\n\n");
      if (!outText) {
        return res.status(204).send("No output from Weka");
      }
      res.type("text/plain").send(outText);
    }
  );
});
checkJava();
app.listen(PORT, () => console.log(`🚀  http://localhost:${PORT}`));
