import bcrypt from "bcryptjs";

export const hashPassword = (plain: string) => bcrypt.hashSync(plain, 10);
export const comparePassword = (plain: string, hash: string) =>
  bcrypt.compareSync(plain, hash);
