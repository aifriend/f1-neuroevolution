import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["node_modules/**", "coverage/**"] },
  js.configs.recommended,
  {
    // Baseline for existing code: surface but don't fail on legacy smells.
    rules: {
      "no-unused-vars": "warn",
      "no-useless-assignment": "warn",
      "no-empty": "warn",
    },
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      globals: { ...globals.browser, ...globals.node },
    },
  },
  {
    files: ["**/*.test.js", "test/**"],
    languageOptions: { globals: { describe: "readonly", it: "readonly", expect: "readonly", beforeEach: "readonly", vi: "readonly" } },
  },
];
