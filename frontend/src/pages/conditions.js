const OPERATOR_SYMBOL = { eq: "=", ne: "≠", gt: ">", gte: "≥", lt: "<", lte: "≤" };

export function describeCondition({ column, operator, value }) {
  if (OPERATOR_SYMBOL[operator]) return `${column} ${OPERATOR_SYMBOL[operator]} ${value}`;
  if (operator === "between") return `${column} antara ${value[0]} dan ${value[1]}`;
  if (operator === "in") return `${column} salah satu dari (${value.join(", ")})`;
  if (operator === "contains") return `${column} mengandung "${value}"`;
  if (operator === "starts_with") return `${column} diawali "${value}"`;
  if (operator === "ends_with") return `${column} diakhiri "${value}"`;
  return `${column} ${operator === "is_null" ? "kosong" : "tidak kosong"}`;
}
