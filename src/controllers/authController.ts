// src/controllers/authController.ts
import express from "express";
import type { Request, Response, NextFunction } from "express";
import jwt from "jsonwebtoken";
import { prisma } from "../db.js";
import { hashPassword, comparePassword } from "../utils/hash.js";

const signToken = (id: number, email: string, role: "USER" | "ADMIN") =>
  jwt.sign({ id, email, role }, process.env.JWT_SECRET!, { expiresIn: "7d" });

export const register = async (req: Request, res: Response): Promise<void> => {
  const { email, password, name, role = "USER" } = req.body;
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
    data: { email, password: hashPassword(password), name, role },
  });

  const token = signToken(user.id, user.email, user.role);
  res
    .cookie("jwt", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 7 * 24 * 60 * 60 * 1000,
    })
    .status(201)
    .json({ message: "Registered!" });
};

export const login = async (req: Request, res: Response): Promise<void> => {
  const { email, password } = req.body;
  const user = await prisma.user.findUnique({ where: { email } });

  if (!user || !comparePassword(password, user.password)) {
    res.status(401).json({ message: "Wrong creds 😵‍💫" });
    return;
  }

  const token = signToken(user.id, user.email, user.role);
  res
    .cookie("jwt", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "none",
      maxAge: 7 * 24 * 60 * 60 * 1000,
    })
    .json({ message: "Logged in" });
};

export const logout = (req: Request, res: Response) : void =>  {
  res.clearCookie("jwt").json({ message: "Logged out" });
};
