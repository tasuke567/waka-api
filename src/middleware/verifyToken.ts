import { RequestHandler } from "express";
import jwt from "jsonwebtoken";

export const verifyToken: RequestHandler = (req, res, next) => {
  const header = req.headers.authorization;          // "Bearer <token>"
  if (!header) {
    res.status(401).json({ message: "No token bro 😢" });
    return;                                          // <- ไม่ return Response
  }

  const [, token] = header.split(" ");
  try {
    req.user = jwt.verify(token, process.env.JWT_SECRET!) as any;
    next();                                          // ผ่าน!
  } catch {
    res.status(403).json({ message: "Invalid token 🫠" });
  }
};
