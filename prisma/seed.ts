import { PrismaClient } from "@prisma/client";
import bcrypt from "bcryptjs";

const prisma = new PrismaClient();

async function main() {
  await prisma.user.create({
    data: {
      email: "admin@example.com",
      name: "Super Admin",
      role: "ADMIN",
      password: bcrypt.hashSync("supersecret", 10),
    },
  });
  console.log("✅  Admin seeded");
}

main()
  .catch((e) => console.error(e))
  .finally(() => prisma.$disconnect());
