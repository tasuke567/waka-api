export function parseWekaStats(out: string) {
    // ดึงเฉพาะบล็อก cross-validation
    const cvBlock = out
      .split("=== Stratified cross-validation ===")[1]      // หลังหัวข้อนี้
      ?.split("=== Detailed Accuracy By Class ===")[0]      // ก่อนหัวข้อถัดไป
      ?? "";
  
    // Accuracy
    const accMatch = cvBlock.match(
      /Correctly\s+Classified\s+Instances\s+(\d+)\s+([\d.]+)\s+%/i
    );
    const accuracy = accMatch ? parseFloat(accMatch[2]) / 100 : null;
  
    // Kappa
    const kappaMatch = cvBlock.match(/Kappa statistic\s+([\d.\-]+)/i);
    const kappa = kappaMatch ? parseFloat(kappaMatch[1]) : null;
  
    return { accuracy, kappa };
  }
  