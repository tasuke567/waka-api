// prisma/seed.ts  (run with:  npx ts-node prisma/seed.ts)
// if you prefer plain JS, change .ts → .mjs, drop type annotations.

import { PrismaClient, Role } from "@prisma/client";
import bcrypt from "bcryptjs";
import { faker } from "@faker-js/faker";

const prisma = new PrismaClient();
const SALT = 10; // bcrypt cost factor
const NUM_USERS = 5; // non-admin accounts

async function createAdmin() {
  await prisma.user.upsert({
    where: { email: "admin@example.com" },
    update: {},
    create: {
      email: "admin@example.com",
      name: "Super Admin",
      role: Role.ADMIN,
      password: bcrypt.hashSync("supersecret", SALT),
    },
  });
}

async function createUsers() {
  for (let i = 0; i < NUM_USERS; i++) {
    const user = await prisma.user.create({
      data: {
        email: faker.internet.email().toLowerCase(),
        name: faker.person.fullName(),
        password: bcrypt.hashSync("password123", SALT),
        // role defaults to USER
      },
    });

    const qCount = faker.number.int({ min: 2, max: 4 });

    for (let j = 0; j < qCount; j++) {
      const questionnaire = await prisma.questionnaire.create({
        data: {
          userId: user.id,
          rawCsvPath: `/uploads/csv/${faker.string.uuid()}.csv`,
        },
      });
      const randProb = () =>
        faker.number.float({ min: 0, max: 1, fractionDigits: 2 });

      // one prediction result per questionnaire
      await prisma.predictionResult.create({
        data: {
          questionnaireId: questionnaire.id,
          label: faker.helpers.arrayElement([
            "Apple",
            "Samsung",
            "Xiaomi",
            "Huawei",
          ]),
          distribution: {
            Apple: randProb(),
            Samsung: randProb(),
            Xiaomi: randProb(),
            Huawei: randProb(),
          },
        },
      });

      // 0-2 feedbacks
      const fbCount = faker.number.int({ min: 0, max: 2 });
      for (let k = 0; k < fbCount; k++) {
        await prisma.feedback.create({
          data: {
            questionnaireId: questionnaire.id,
            uiEase: faker.number.int({ min: 1, max: 5 }),
            satisfaction: faker.number.int({ min: 1, max: 5 }),
            clarity: faker.number.int({ min: 1, max: 5 }),
          },
        });
      }
    }
  }
}

async function main() {
  console.time("🌱  Seeding");
  await createAdmin();
  await createUsers();
  console.timeEnd("🌱  Seeding");
  console.log("✅  Database seeded successfully!");
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
