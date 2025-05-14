import express from "express";
import type { RequestHandler, Request, Response, NextFunction } from "express";

import jwt from "jsonwebtoken";

export const verifyToken: RequestHandler = (req, res, next) => {
  const auth = req.headers.authorization;
  if (!auth?.startsWith("Bearer ")) {
    res.status(401).json({ error: "Missing token" });
    return;                // ⬅️  return void
  }

  try {
    const payload = jwt.verify(
      auth.slice(7),
      process.env.JWT_SECRET!
    ) as { id: number; email: string; role: "USER" | "ADMIN" };

    req.user = { id: payload.id, email: payload.email, role: payload.role };
    next();
  } catch {
    res.status(401).json({ error: "Invalid token" });
  }
};
