/* ----- utils/parseWeka.ts ----------------------------------------- */
export function parseWekaStats(stdout: string) {
    let accuracy: number | null = null;
    let kappa:    number | null = null;
  
    stdout.split("\n").forEach((l) => {
      // ex. "Correctly Classified Instances          95               95      95.0000 %"
      if (/Correctly\s+Classified\s+Instances/i.test(l)) {
        const parts = l.trim().split(/\s+/);
        const last  = parts.at(-2) ?? "";      // คอลัมน์ก่อนเครื่องหมาย %
        accuracy = parseFloat(last);           // 95.0000
      }
      // ex. "Kappa statistic                          0.72"
      if (/Kappa statistic/i.test(l)) {
        const val = l.trim().split(/\s+/).at(-1);
        kappa = val ? parseFloat(val) : null;
      }
    });
  
    return { accuracy, kappa };
  }
  