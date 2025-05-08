import { RequestHandler } from "express";
import jwt from "jsonwebtoken";
import { prisma } from "../db";
import { hashPassword, comparePassword } from "../utils/hash";

const signToken = (id: number, email: string) =>
  jwt.sign({ id, email }, process.env.JWT_SECRET!, { expiresIn: "7d" });

export const register: RequestHandler = async (req, res) => {
  const { email, password, name } = req.body;
  if (!email || !password) {
    res.status(400).json({ message: "email / password required" });
    return;
  }

  const exists = await prisma.user.findUnique({ where: { email } });
  if (exists) {
    res.status(409).json({ message: "Email taken 🥲" });
    return;
  }

  const user = await prisma.user.create({
    data: { email, password: hashPassword(password), name },
  });
  res.status(201).json({ token: signToken(user.id, user.email) });
};

export const login: RequestHandler = async (req, res) => {
  const { email, password } = req.body;
  const user = await prisma.user.findUnique({ where: { email } });
  if (!user || !comparePassword(password, user.password)) {
    res.status(401).json({ message: "Wrong creds 😵‍💫" });
    return;
  }
  res.json({ token: signToken(user.id, user.email) });
};
