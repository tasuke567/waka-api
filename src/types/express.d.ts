import "express-serve-static-core";

declare module "express-serve-static-core" {
  interface Request {
    /** inject จาก middleware verifyToken */
    user?: { id: number; email: string };
  }
}
