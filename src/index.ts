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
const MTJ_JAR =
  process.env.MTJ_JAR ?? path.join(__dirname, "../model/mtj-1.0.4.jar");
const MODEL = path.join(__dirname, "../model/myJ48.model");
const HEADER_PATH = path.join(__dirname, "../model/header.arff");
const HEADER = existsSync(HEADER_PATH)
  ? fs.readFileSync(HEADER_PATH, "utf8")
  : fs.readFileSync(path.join(__dirname, "../model/header.arff.tpl"), "utf8");
const CLASS_ATTR = "Current_brand"; // ชื่อคอลัมน์ที่เป็น class label
const WEKA_CP = [WEKA_JAR, MTJ_JAR].join(path.delimiter);

interface Prediction {
  label: string; // เช่น "Apple"
  distribution: number[]; // เช่น [0.10, 0.05, 0.40, 0.15, 0.30]
}

// Replace checkJava() with this version
function checkJava() {
  const javaPath = path.join(process.cwd(), "java/bin/java");
  const minVersion = 17;

  try {
    const versionOutput = execSync(`"${javaPath}" -version 2>&1`).toString();
    const versionMatch = versionOutput.match(/version "(\d+)\./);
    
    if (!versionMatch || parseInt(versionMatch[1]) < minVersion) {
      throw new Error(`Java ${minVersion}+ required`);
    }
  } catch (e) {
    throw new Error(`Java check failed: ${e instanceof Error ? e.message : String(e)}`);
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
async function buildArff(
  csvPath: string,
  isTrain: boolean,
  modelDir: string
): Promise<string> {
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
          mapValues: ({ value }) =>
            typeof value === "string" ? value.trim().normalize("NFKC") : value, // เพิ่มการ normalize ค่า
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
  const headerPath = path.join(modelDir, "header.arff");
  const HEADER = fs.readFileSync(headerPath, "utf8");

  // 3) Extract attributes
  const cols = HEADER.split("\n")
    .filter((l) => l.trim().startsWith("@ATTRIBUTE"))
    .map((l) => l.trim().split(/\s+/)[1]);
  // ในฟังก์ชัน buildArff
  if (isTrain && !cols.includes(CLASS_ATTR)) {
    throw new Error(`Missing class attribute "${CLASS_ATTR}" in CSV`);
  }
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
    const javaPath = path.resolve(process.cwd(), "java/bin/java");
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
    console.log("Executing Weka with:", [javaPath, ...args].join(" "));
    execFile(javaPath, args, { encoding: "utf8" }, (err, stdout, stderr) => {
      console.log("=== WEKA STDOUT ===\n", stdout);
      console.log("=== WEKA STDERR ===\n", stderr);

      if (err || stderr.includes("Exception")) {
        return reject(
          new Error(stderr || err?.message || "Weka execution failed")
        );
      }

      // Improved parsing logic
      try {
        const predictionLine = stdout
          .split("\n")
          .find((line) => line.includes(":") && line.includes("distribution"));

        if (!predictionLine) {
          throw new Error("No prediction line found");
        }

        const parts = predictionLine
          .trim()
          .split(/\s+/)
          .filter((p) => p !== "");

        const label = parts[parts.length - 2].split(":").pop()!;
        const distribution = parts[parts.length - 1]
          .replace(/[\[\]]/g, "")
          .split(",")
          .map(parseFloat);

        resolve({ label, distribution });
      } catch (parseError) {
        reject(
          new Error(
            `Failed to parse Weka output: ${parseError}\nOutput:\n${stdout}`
          )
        );
      }
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
    const javaPath = path.resolve(process.cwd(), "java/bin/java");
    // สร้างโฟลเดอร์เก็บโมเดลถ้ายังไม่มี
    const modelDir = path.dirname(MODEL);
    const headerPath = path.join(modelDir, "header.arff");
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

    console.log("Training CMD:", [javaPath, ...args].join(" "));
    execFile(javaPath, args, { encoding: "utf8" }, (e, out, err) => {
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

  const tmp = await buildArff(req.file.path, false, path.dirname(MODEL)); // isTrain = false

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

    const filesToDelete = [
      req.file?.path,
      tmp,
      path.join(uploadDir, "empty.arff"),
    ];

    await Promise.all(
      filesToDelete
        .filter(Boolean)
        .map((file) => f.unlink(file!).catch(console.error))
    );
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
        : await buildArff(req.file.path, true, path.dirname(MODEL)); // isTrain = true
    const actualHeader = fs
      .readFileSync(arffPath, "utf8")
      .split("@DATA")[0]
      .trim();

    const modelDir = path.dirname(MODEL);
    fs.writeFileSync(path.join(modelDir, "header.arff"), actualHeader);
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
// เพิ่มก่อน app.listen()
const requiredFiles = [
  path.join(__dirname, "../model/header.arff"),
  path.join(__dirname, "../model/header.arff.tpl"),
  MODEL
];

requiredFiles.forEach(file => {
  if (!existsSync(file)) {
    throw new Error(`Missing required file: ${file}`);
  }
});
app.listen(PORT, () => console.log(`🚀  http://localhost:${PORT}`));
