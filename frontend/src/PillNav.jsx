import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import './PillNav.css'

function PillNav({
  items,
  activeHref,
  className = '',
  ease = 'power3.easeOut',
  baseColor = '#ffffff',
  pillColor = '#060010',
  hoveredPillTextColor = '#060010',
  pillTextColor,
  initialLoadAnimation = true,
  onItemSelect,
}) {
  const resolvedPillTextColor = pillTextColor ?? baseColor
  const circleRefs = useRef([])
  const tlRefs = useRef([])
  const activeTweenRefs = useRef([])
  const navItemsRef = useRef(null)

  useEffect(() => {
    const layout = () => {
      circleRefs.current.forEach((circle, i) => {
        if (!circle?.parentElement) return

        const pill = circle.parentElement
        const rect = pill.getBoundingClientRect()
        const { width: w, height: h } = rect
        const R = ((w * w) / 4 + h * h) / (2 * h)
        const D = Math.ceil(2 * R) + 2
        const delta = Math.ceil(R - Math.sqrt(Math.max(0, R * R - (w * w) / 4))) + 1
        const originY = D - delta

        circle.style.width = `${D}px`
        circle.style.height = `${D}px`
        circle.style.bottom = `-${delta}px`

        gsap.set(circle, {
          xPercent: -50,
          scale: 0,
          transformOrigin: `50% ${originY}px`,
        })

        const label = pill.querySelector('.pill-label')
        const white = pill.querySelector('.pill-label-hover')

        if (label) gsap.set(label, { y: 0 })
        if (white) gsap.set(white, { y: h + 12, opacity: 0 })

        tlRefs.current[i]?.kill()
        const tl = gsap.timeline({ paused: true })

        tl.to(circle, { scale: 1.2, xPercent: -50, duration: 2, ease, overwrite: 'auto' }, 0)

        if (label) {
          tl.to(label, { y: -(h + 8), duration: 2, ease, overwrite: 'auto' }, 0)
        }

        if (white) {
          gsap.set(white, { y: Math.ceil(h + 100), opacity: 0 })
          tl.to(white, { y: 0, opacity: 1, duration: 2, ease, overwrite: 'auto' }, 0)
        }

        tlRefs.current[i] = tl
      })
    }

    layout()

    const onResize = () => layout()
    window.addEventListener('resize', onResize)

    if (document.fonts?.ready) {
      document.fonts.ready.then(layout).catch(() => {})
    }

    if (initialLoadAnimation) {
      const navItems = navItemsRef.current
      if (navItems) {
        gsap.set(navItems, { width: 0, overflow: 'hidden' })
        gsap.to(navItems, {
          width: 'auto',
          duration: 0.6,
          ease,
        })
      }
    }

    return () => window.removeEventListener('resize', onResize)
  }, [items, ease, initialLoadAnimation])

  const handleEnter = (i) => {
    const tl = tlRefs.current[i]
    if (!tl) return
    activeTweenRefs.current[i]?.kill()
    activeTweenRefs.current[i] = tl.tweenTo(tl.duration(), {
      duration: 0.3,
      ease,
      overwrite: 'auto',
    })
  }

  const handleLeave = (i) => {
    const tl = tlRefs.current[i]
    if (!tl) return
    activeTweenRefs.current[i]?.kill()
    activeTweenRefs.current[i] = tl.tweenTo(0, {
      duration: 0.2,
      ease,
      overwrite: 'auto',
    })
  }

  const cssVars = {
    ['--base']: baseColor,
    ['--pill-bg']: pillColor,
    ['--hover-text']: hoveredPillTextColor,
    ['--pill-text']: resolvedPillTextColor,
  }

  return (
    <div className="pill-nav-container">
      <nav className={`pill-nav ${className}`} aria-label="Study Module Switcher" style={cssVars}>
        <div className="pill-nav-items" ref={navItemsRef}>
          <ul className="pill-list" role="tablist">
            {items.map((item, i) => (
              <li key={item.href || `item-${i}`} role="presentation">
                <button
                  role="tab"
                  aria-selected={activeHref === item.href}
                  className={`pill${activeHref === item.href ? ' is-active' : ''}`}
                  aria-label={item.ariaLabel || item.label}
                  onMouseEnter={() => handleEnter(i)}
                  onMouseLeave={() => handleLeave(i)}
                  onClick={() => onItemSelect?.(item)}
                  type="button"
                >
                  <span
                    className="hover-circle"
                    aria-hidden="true"
                    ref={(el) => {
                      circleRefs.current[i] = el
                    }}
                  />
                  <span className="label-stack">
                    <span className="pill-label">{item.label}</span>
                    <span className="pill-label-hover" aria-hidden="true">
                      {item.label}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </nav>
    </div>
  )
}

export default PillNav
