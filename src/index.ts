// Express + Weka API (Cleaned + Ready for Deploy)

import crypto from "node:crypto";
import express from "express";
import multer from "multer";
import { execFile, execSync } from "node:child_process";
import fs, { promises as fsp } from "node:fs";
import { existsSync, mkdirSync, statSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "url";
import csvParser from "csv-parser";
import { v4 as uuidv4 } from "uuid";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ?? 3000;
const MODEL = path.join(__dirname, "../model/myJ48.model");
const HEADER_PATH = path.join(__dirname, "../model/header.arff");
const HEADER = existsSync(HEADER_PATH)
  ? fs.readFileSync(HEADER_PATH, "utf8")
  : fs.readFileSync(path.join(__dirname, "../model/header.arff.tpl"), "utf8");
const WEKA_JAR = path.join(__dirname, "../model/weka.jar");
const MTJ_JAR = path.join(__dirname, "../model/mtj-1.0.4.jar");
const WEKA_CP = [WEKA_JAR, MTJ_JAR].join(path.delimiter);
const CLASS_ATTR = "Current_brand";
const uploadDir = path.join(__dirname, "uploads");
const trainDir = path.join(uploadDir, "train");
const isWin = process.platform === "win32";
const javaPath = path.join(process.cwd(), "java/bin/java");

// Ensure directories
[uploadDir, trainDir].forEach((dir) => {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
});
function checkJava() {
  try {
    execSync(`${javaPath} -version`);
    console.log("Java found ✅ (version check skipped)");
  } catch (e) {
    throw new Error(
      "Java check failed: " + (e instanceof Error ? e.message : String(e))
    );
  }
}

const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, uploadDir),
  filename: (_, file, cb) =>
    cb(null, `${crypto.randomUUID()}${path.extname(file.originalname)}`),
});

const upload = multer({
  storage,
  limits: { fileSize: 5_000_000 },
  fileFilter: (_, file, cb) => {
    const allowed = [
      "text/csv",
      "text/plain",
      "application/octet-stream",
      "application/x-arff",
    ];
    cb(null, allowed.includes(file.mimetype));
  },
});

async function buildArff(csvPath: string, isTrain: boolean): Promise<string> {
  const rows: Record<string, string>[] = [];
  await new Promise<void>((res, rej) => {
    fs.createReadStream(csvPath)
      .pipe(
        csvParser({
          mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim(),
          mapValues: ({ value }) => value.trim().normalize("NFKC"),
        })
      )
      .on("data", (row) => rows.push(row))
      .on("end", res)
      .on("error", rej);
  });

  const headerText = fs.readFileSync(HEADER_PATH, "utf8");
  const cols = headerText
    .split("\n")
    .filter((l) => l.startsWith("@ATTRIBUTE"))
    .map((l) => l.split(/\s+/)[1]);

  const arffPath = path.join(uploadDir, `${uuidv4()}.arff`);
  await new Promise<void>((resolve, reject) => {
    const ws = fs.createWriteStream(arffPath);
    ws.on("error", reject).on("finish", resolve);
    ws.write(headerText.trim() + "\n@DATA\n");
    for (const row of rows) {
      const line = cols
        .map((col) => {
          const val = col === CLASS_ATTR && !isTrain ? "?" : row[col];
          return /[\s,{}]/.test(val) ? `'${val}'` : val;
        })
        .join(",");
      ws.write(line + "\n");
    }
    ws.end();
  });
  return arffPath;
}

function wekaPredict(arff: string, model: string): Promise<any> {
  const javaPath = path.join(process.cwd(), "java/bin/java");
  const args = [
    "-Xmx1G",
    "-cp",
    WEKA_CP,
    "weka.classifiers.trees.J48",
    "-l",
    model,
    "-T",
    arff.replace(/\\/g, "/"),
    "-p",
    "0",
    "-distribution",
  ];
  return new Promise((resolve, reject) => {
    execFile(javaPath, args, { encoding: "utf8" }, (err, stdout, stderr) => {
      if (err || stderr.includes("Exception")) {
        const msg = [
          "Weka Execution Error:",
          `Args: ${args.join(" ")}`,
          `stderr:\n${stderr}`,
          `stdout:\n${stdout}`,
        ].join("\n\n");

        console.error(msg);
        return reject(new Error(msg));
      }
      if (stdout.includes("No training set")) {
        return reject("No training set found in the model.");
      }
      if (stdout.includes("No test set")) {
        const line = stdout.split("\n").find((l) => l.includes("distribution"));
        if (!line) return reject("No prediction line found");
        const parts = line.trim().split(/\s+/);
        resolve({
          label: parts.at(-2)?.split(":").pop(),
          distribution: parts
            .at(-1)
            ?.replace(/[\[\]]/g, "")
            .split(",")
            .map(Number),
        });
      }
    });
  });
}

const app = express();

app.post("/predict", upload.single("file"), async (req, res) => {
  try {
    const arffPath = await buildArff(req.file!.path, false);
    const fileName = `predict-${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}-${uuidv4()}.arff`;
    const finalPath = path.join(uploadDir, fileName);
    fs.copyFileSync(arffPath, finalPath);
    const prediction = await wekaPredict(finalPath, MODEL);
    res.json({ prediction });
  } catch (e) {
    const errorMessage = `Prediction failed: ${String(e)}`;
    console.error(errorMessage);
    res.status(500).json({
      error: "Weka error",
      message: errorMessage,
      stack: e instanceof Error ? e.stack : undefined,
    });
  }
});

app.post("/train", upload.single("file"), async (req, res) => {
  try {
    const arffPath = await buildArff(req.file!.path, true);
    const fileName = `train-${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}-${uuidv4()}.arff`;
    const finalPath = path.join(trainDir, fileName);
    fs.copyFileSync(arffPath, finalPath);
    const header = fs.readFileSync(arffPath, "utf8").split("@DATA")[0];
    fs.writeFileSync(HEADER_PATH, header);
    res.json({ saved: finalPath });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.get("/predict-history", (req, res) => {
  const files = readdirSync(uploadDir)
    .filter((f) => f.startsWith("predict-") && f.endsWith(".arff"))
    .map((f) => ({
      file: f,
      time: statSync(path.join(uploadDir, f)).birthtime,
    }));
  res.json(files);
});

app.get("/train-history", (req, res) => {
  const files = readdirSync(trainDir)
    .filter((f) => f.startsWith("train-") && f.endsWith(".arff"))
    .map((f) => ({
      file: f,
      time: statSync(path.join(trainDir, f)).birthtime,
    }));
  res.json(files);
});

checkJava();
app.listen(PORT, () => console.log(`🚀  http://localhost:${PORT}`));
