// The prompt the research team starts from, shown on the home page: the opening of
// RESEARCHER_RULES in app/agents/prompts.py and the task line of the dataset card
// (data/churn/meta.json). tests/test_web_prompt.py keeps them in sync; change both together.
export const INITIAL_PROMPT = 'You are an ML researcher on a small team. Your goal: lower the validation log loss of the model in train.py.'
export const TASK = 'Predict the chance (0 to 1) that a bank customer leaves.'
export const DATASET = { name: 'churn', source: 'OpenML 46371', trainRows: 24499 }
