import { useEffect, useMemo, useState } from 'react'
import { QUIZ_PASS_SCORE } from './quizStorage'
import { getQuizQuestions } from './quizBank'

function QuizProgress({ currentIndex, total }) {
  const percent = total > 0 ? ((currentIndex + 1) / total) * 100 : 0
  return (
    <div className="quiz-progress-wrap" aria-label="Quiz progress">
      <div className="quiz-progress-meta">
        <span>Question {currentIndex + 1} of {total}</span>
        <span>{Math.round(percent)}%</span>
      </div>
      <div className="quiz-progress-track">
        <div className="quiz-progress-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  )
}

function OptionButton({ option, selected, correct, locked, onSelect }) {
  const className = [
    'quiz-option',
    selected ? 'selected' : '',
    locked && correct ? 'correct' : '',
    locked && selected && !correct ? 'incorrect' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button type="button" className={className} onClick={onSelect} disabled={locked}>
      <span className="quiz-option-marker">{selected ? '●' : '○'}</span>
      <span>{option}</span>
    </button>
  )
}

export default function QuizPage({ moduleName, moduleDescription, onBackToModule, onFinishQuiz }) {
  const questions = useMemo(() => getQuizQuestions(moduleName), [moduleName])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [answers, setAnswers] = useState([])
  const [submitted, setSubmitted] = useState(false)
  const [score, setScore] = useState(0)

  useEffect(() => {
    setCurrentIndex(0)
    setAnswers(Array.from({ length: questions.length }, () => ''))
    setSubmitted(false)
    setScore(0)
  }, [moduleName, questions.length])

  const currentQuestion = questions[currentIndex]
  const answeredCount = answers.filter(Boolean).length
  const allAnswered = questions.length > 0 && answeredCount === questions.length
  const currentAnswer = answers[currentIndex] || ''
  const pass = submitted && score >= QUIZ_PASS_SCORE

  const handleSelect = (option) => {
    setAnswers((previous) => {
      const next = previous.slice()
      next[currentIndex] = option
      return next
    })
  }

  const handleSubmit = () => {
    if (!allAnswered || submitted) return
    const nextScore = questions.reduce((total, question, index) => {
      return total + (answers[index] === question.answer ? 1 : 0)
    }, 0)
    setScore(nextScore)
    setSubmitted(true)
    onFinishQuiz?.({
      module: moduleName,
      score: nextScore,
      total: questions.length,
      completed: nextScore >= QUIZ_PASS_SCORE,
    })
  }

  if (!questions.length) {
    return (
      <section className="quiz-page-shell">
        <div className="quiz-card quiz-empty-card">
          <div className="quiz-page-head">
            <div>
              <p className="quiz-kicker">Module assessment</p>
              <h1>{moduleName}</h1>
              {moduleDescription ? <p className="quiz-description">{moduleDescription}</p> : null}
            </div>
            <span className="badge">No quiz yet</span>
          </div>
          <p className="module-note">This module does not have a quiz bank configured yet.</p>
          <button type="button" className="module-action-btn primary" onClick={onBackToModule}>
            Return to module
          </button>
        </div>
      </section>
    )
  }

  return (
    <section className="quiz-page-shell">
      <div className="quiz-card">
        <div className="quiz-page-head">
          <div>
            <p className="quiz-kicker">Module assessment</p>
            <h1>{moduleName}</h1>
            {moduleDescription ? <p className="quiz-description">{moduleDescription}</p> : null}
          </div>
          <div className="quiz-head-badges">
            <span className={submitted ? (pass ? 'badge success' : 'badge warning') : 'badge'}>
              {submitted ? (pass ? 'Passed' : 'Retry available') : '20 questions'}
            </span>
            <span className="badge">Pass: 16/20</span>
          </div>
        </div>

        {!submitted ? (
          <>
            <QuizProgress currentIndex={currentIndex} total={questions.length} />
            <div className="quiz-question-card">
              <p className="quiz-question-index">Question {currentIndex + 1}</p>
              <h2>{currentQuestion.question}</h2>
              <div className="quiz-options-grid">
                {currentQuestion.options.map((option) => (
                  <OptionButton
                    key={option}
                    option={option}
                    selected={currentAnswer === option}
                    correct={option === currentQuestion.answer}
                    locked={submitted}
                    onSelect={() => handleSelect(option)}
                  />
                ))}
              </div>
            </div>

            <div className="quiz-actions-row">
              <button
                type="button"
                className="module-action-btn"
                onClick={() => setCurrentIndex((value) => Math.max(0, value - 1))}
                disabled={currentIndex === 0}
              >
                Previous
              </button>
              <div className="quiz-actions-spacer" />
              {currentIndex < questions.length - 1 ? (
                <button
                  type="button"
                  className="module-action-btn primary"
                  onClick={() => setCurrentIndex((value) => Math.min(questions.length - 1, value + 1))}
                >
                  Next
                </button>
              ) : (
                <button type="button" className="module-action-btn primary" onClick={handleSubmit} disabled={!allAnswered}>
                  Submit Quiz
                </button>
              )}
            </div>
            <p className="quiz-helper-text">
              Answer all 20 questions before submitting. You can move back and forth to review your choices.
            </p>
          </>
        ) : (
          <>
            <div className={pass ? 'quiz-result-banner pass' : 'quiz-result-banner fail'}>
              <div>
                <p className="quiz-result-label">Final score</p>
                <h2>
                  {score}/{questions.length}
                </h2>
              </div>
              <div className="quiz-result-copy">
                <p>{pass ? 'Module completed.' : 'You did not reach the pass mark yet.'}</p>
                <p>{pass ? 'The module is now marked complete.' : 'Retake the quiz after reviewing the explanations below.'}</p>
              </div>
            </div>

            <div className="quiz-review-list">
              {questions.map((question, index) => {
                const answer = answers[index]
                const isCorrect = answer === question.answer
                return (
                  <article key={question.id} className={isCorrect ? 'quiz-review-item correct' : 'quiz-review-item incorrect'}>
                    <div className="quiz-review-head">
                      <span>Q{index + 1}</span>
                      <span>{isCorrect ? 'Correct' : 'Review'}</span>
                    </div>
                    <h3>{question.question}</h3>
                    <p className="quiz-review-answer">Your answer: {answer || 'No answer selected'}</p>
                    <p className="quiz-review-answer">Correct answer: {question.answer}</p>
                    <p className="quiz-review-explanation">{question.explanation}</p>
                  </article>
                )
              })}
            </div>

            <div className="quiz-actions-row quiz-results-actions">
              <button type="button" className="module-action-btn" onClick={onBackToModule}>
                Return to module
              </button>
              <button type="button" className="module-action-btn primary" onClick={() => setSubmitted(false)}>
                Retry Quiz
              </button>
            </div>
          </>
        )}
      </div>
    </section>
  )
}
