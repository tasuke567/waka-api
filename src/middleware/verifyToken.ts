import type { Request, Response, NextFunction } from "express";
import jwt from "jsonwebtoken";

export const verifyToken = (req: Request, res: Response, next: NextFunction): void => {
  const token = req.cookies?.jwt;

  if (!token) {
    res.status(401).json({ error: "Missing token" });
    return;
  }

  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET!) as {
      id: number;
      email: string;
      role: "USER" | "ADMIN";
    };

    req.user = {
      id: payload.id,
      email: payload.email,
      role: payload.role,
    };

    next();
  } catch {
    res.status(401).json({ error: "Invalid token" });
  }
};
