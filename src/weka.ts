import { exec } from 'child_process';
import path from 'path';

export function trainModel(csvPath: string, modelPath: string, algorithm: string): Promise<void> {
  const algoMap: Record<string, string> = {
    J48: 'weka.classifiers.trees.J48',
    NaiveBayes: 'weka.classifiers.bayes.NaiveBayes',
  };

  const algoClass = algoMap[algorithm] || algoMap['J48'];
  const command = `java -cp weka.jar ${algoClass} -t ${csvPath} -d ${modelPath}`;

  return new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`🔥 Error: ${stderr}`);
        reject(error);
      } else {
        console.log(`✅ Model trained with ${algorithm}:\n${stdout}`);
        resolve();
      }
    });
  });
}
