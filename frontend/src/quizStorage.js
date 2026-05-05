const QUIZ_PROGRESS_STORAGE_KEY = 'deepnexus.quiz.progress.v1'
const QUIZ_USER_ID = 'local-user'
const QUIZ_PASS_SCORE = 16

function readLocalStorageJson(key, fallback) {
  if (typeof window === 'undefined') return fallback
  try {
    const raw = window.localStorage.getItem(key)
    if (!raw) return fallback
    return JSON.parse(raw)
  } catch {
    return fallback
  }
}

function writeLocalStorageJson(key, value) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(key, JSON.stringify(value))
}

export function loadQuizProgress() {
  const parsed = readLocalStorageJson(QUIZ_PROGRESS_STORAGE_KEY, [])
  return Array.isArray(parsed) ? parsed : []
}

export function saveQuizProgress(records) {
  writeLocalStorageJson(QUIZ_PROGRESS_STORAGE_KEY, records)
}

export function upsertQuizProgress(record) {
  const records = loadQuizProgress()
  const userId = typeof record?.user_id === 'string' && record.user_id.trim() ? record.user_id.trim() : QUIZ_USER_ID
  const nextRecord = {
    user_id: userId,
    module: record.module,
    score: record.score,
    rawScore: record.rawScore,
    total: record.total,
    completed: Boolean(record.completed),
    updatedAt: Date.now(),
  }

  const existingIndex = records.findIndex((item) => item.module === nextRecord.module && item.user_id === userId)
  if (existingIndex >= 0) {
    const existing = records[existingIndex]
    records[existingIndex] = {
      ...existing,
      ...nextRecord,
      completed: Boolean(existing.completed || nextRecord.completed),
    }
  } else {
    records.push(nextRecord)
  }

  saveQuizProgress(records)
  return records
}

export function getQuizProgressMap(records = loadQuizProgress()) {
  return records.reduce((accumulator, record) => {
    accumulator[record.module] = record
    return accumulator
  }, {})
}

export { QUIZ_PASS_SCORE, QUIZ_USER_ID, QUIZ_PROGRESS_STORAGE_KEY }
