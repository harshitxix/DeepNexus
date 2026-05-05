import { useEffect, useMemo, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

import './ScrollReveal.css'

gsap.registerPlugin(ScrollTrigger)

function ScrollReveal({
  children,
  scrollContainerRef,
  enableBlur = true,
  baseOpacity = 0.14,
  baseRotation = 0,
  blurStrength = 3,
  containerClassName = '',
  textClassName = '',
  rotationEnd = 'bottom bottom',
  wordAnimationEnd = 'bottom bottom',
  as = 'div',
  textAs = 'p',
}) {
  const containerRef = useRef(null)

  const splitText = useMemo(() => {
    const text = typeof children === 'string' ? children : ''
    return text.split(/(\s+)/).map((word, index) => {
      if (/^\s+$/.test(word)) return word
      return (
        <span className="word" key={index}>
          {word}
        </span>
      )
    })
  }, [children])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return undefined

    const scroller = scrollContainerRef?.current || null

    const triggerBase = {
      trigger: el,
      scrub: 0.9,
    }

    if (scroller) {
      triggerBase.scroller = scroller
    }

    const ctx = gsap.context(() => {
      // rotation animation for the container
      gsap.fromTo(
        el,
        { transformOrigin: '0% 50%', rotate: baseRotation },
        {
          ease: 'none',
          rotate: 0,
          scrollTrigger: {
            ...triggerBase,
            start: 'top bottom',
            end: rotationEnd,
          },
        }
      )

      // always animate container opacity + blur as a fallback so non-string children are handled
      if (enableBlur) {
        gsap.fromTo(
          el,
          { opacity: baseOpacity, filter: `blur(${blurStrength}px)` },
          {
            ease: 'none',
            opacity: 1,
            filter: 'blur(0px)',
            scrollTrigger: {
              ...triggerBase,
              start: 'top bottom-=15%',
              end: wordAnimationEnd,
            },
          }
        )
      } else {
        // if blur not enabled ensure opacity transition still occurs
        gsap.fromTo(
          el,
          { opacity: baseOpacity },
          {
            ease: 'none',
            opacity: 1,
            scrollTrigger: {
              ...triggerBase,
              start: 'top bottom-=15%',
              end: wordAnimationEnd,
            },
          }
        )
      }

      // word-level animations only when children is a plain string (we split into .word spans)
      const wordElements = el.querySelectorAll('.word')
      if (wordElements && wordElements.length > 0) {
        gsap.fromTo(
          wordElements,
          { opacity: baseOpacity, willChange: 'opacity, filter' },
          {
            ease: 'none',
            opacity: 1,
            stagger: 0.035,
            scrollTrigger: {
              ...triggerBase,
              start: 'top bottom-=15%',
              end: wordAnimationEnd,
            },
          }
        )

        if (enableBlur) {
          gsap.fromTo(
            wordElements,
            { filter: `blur(${blurStrength}px)` },
            {
              ease: 'none',
              filter: 'blur(0px)',
              stagger: 0.035,
              scrollTrigger: {
                ...triggerBase,
                start: 'top bottom-=15%',
                end: wordAnimationEnd,
              },
            }
          )
        }
      }
    }, el)

    // Ensure ScrollTrigger recalculates when this element's size changes (e.g., tab show/hide)
    let ro = null
    try {
      ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          // when element becomes visible/has height, refresh ScrollTrigger to recalc positions
          if (entry.contentRect && entry.contentRect.height > 0) {
            if (gsap && gsap.core && gsap.core.globals && gsap.core.globals.ScrollTrigger) {
              gsap.core.globals.ScrollTrigger.refresh()
            }
          }
        }
      })
      ro.observe(el)
    } catch (e) {
      // ResizeObserver may not be available in some environments; fallback to a delayed refresh
      setTimeout(() => {
        if (gsap && gsap.core && gsap.core.globals && gsap.core.globals.ScrollTrigger) {
          gsap.core.globals.ScrollTrigger.refresh()
        }
      }, 120)
    }

    return () => {
      if (ro && ro.disconnect) ro.disconnect()
      ctx.revert()
    }
  }, [scrollContainerRef, enableBlur, baseRotation, baseOpacity, rotationEnd, wordAnimationEnd, blurStrength, children])

  const Tag = as
  const TextTag = textAs

  return (
    <Tag ref={containerRef} className={`scroll-reveal ${containerClassName}`.trim()}>
      <TextTag className={`scroll-reveal-text ${textClassName}`.trim()}>{typeof children === 'string' ? splitText : children}</TextTag>
    </Tag>
  )
}

export default ScrollReveal
