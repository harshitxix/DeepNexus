import React, { useCallback, useLayoutEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import './StaggeredMenu.css'
import VariableProximity from './VariableProximity'

function StaggeredMenu({
  items = [],
  isOpen = false,
  onToggle,
  onItemClick,
  className = '',
  toggleVariant = 'default',
  activeItemKey = null,
}) {
  const panelRef = useRef(null)
  const preLayersRef = useRef(null)
  const preLayerElsRef = useRef([])
  const itemEntranceTweenRef = useRef(null)
  const openTlRef = useRef(null)
  const closeTweenRef = useRef(null)
  const itemLabelsRef = useRef([])

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      const panel = panelRef.current
      const preContainer = preLayersRef.current
      if (!panel || !preContainer) return

      const preLayers = Array.from(preContainer.querySelectorAll('.sm-prelayer'))
      preLayerElsRef.current = preLayers

      gsap.set([panel, ...preLayers], { xPercent: -100 })
      gsap.set(itemLabelsRef.current, { yPercent: 120, opacity: 0 })
    })

    return () => ctx.revert()
  }, [])

  const playOpen = useCallback(() => {
    const panel = panelRef.current
    const layers = preLayerElsRef.current
    if (!panel) return

    openTlRef.current?.kill()
    closeTweenRef.current?.kill()
    itemEntranceTweenRef.current?.kill()

    const tl = gsap.timeline()

    layers.forEach((layer, i) => {
      tl.to(
        layer,
        {
          xPercent: 0,
          duration: 0.45,
          ease: 'power4.out',
        },
        i * 0.08
      )
    })

    tl.to(
      panel,
      {
        xPercent: 0,
        duration: 0.58,
        ease: 'power4.out',
      },
      0.12
    )

    itemEntranceTweenRef.current = gsap.to(itemLabelsRef.current, {
      yPercent: 0,
      opacity: 1,
      duration: 0.6,
      ease: 'power3.out',
      stagger: 0.08,
      delay: 0.18,
    })

    openTlRef.current = tl
  }, [])

  const playClose = useCallback(() => {
    const panel = panelRef.current
    const layers = preLayerElsRef.current
    if (!panel) return

    openTlRef.current?.kill()
    itemEntranceTweenRef.current?.kill()
    gsap.set(itemLabelsRef.current, { yPercent: 120, opacity: 0 })

    closeTweenRef.current = gsap.to([panel, ...layers], {
      xPercent: -100,
      duration: 0.32,
      ease: 'power3.in',
      overwrite: 'auto',
    })
  }, [])

  React.useEffect(() => {
    if (isOpen) {
      playOpen()
    } else {
      playClose()
    }
  }, [isOpen, playOpen, playClose])

  return (
    <div className={`staggered-menu-wrapper ${className}`} data-open={isOpen || undefined}>
      <div ref={preLayersRef} className="sm-prelayers" aria-hidden="true">
        <div className="sm-prelayer" style={{ background: '#b19eef' }} />
        <div className="sm-prelayer" style={{ background: '#5227ff' }} />
      </div>

      <button
        className={toggleVariant === 'triangle' ? 'sm-toggle sm-toggle-triangleMode' : 'sm-toggle'}
        type="button"
        aria-expanded={isOpen}
        aria-label={isOpen ? 'Close menu' : 'Open menu'}
        onClick={onToggle}
      >
        {toggleVariant === 'triangle' ? (
          <span className={`sm-toggle-triangle${isOpen ? ' open' : ''}`} aria-hidden="true" />
        ) : (
          <>
            <span>{isOpen ? 'Close' : 'Menu'}</span>
            <span className={`sm-toggle-icon${isOpen ? ' open' : ''}`}>+</span>
          </>
        )}
      </button>

      <aside ref={panelRef} className="staggered-menu-panel" aria-hidden={!isOpen}>
        <div className="sm-panel-inner">
          <ul className="sm-panel-list" role="list">
            {items.map((item, idx) => (
              <li className="sm-panel-itemWrap" key={item.label + idx}>
                <button
                  type="button"
                  className={activeItemKey === item.moduleName ? 'sm-panel-item active' : 'sm-panel-item'}
                  onClick={() => onItemClick?.(item)}
                >
                  <span
                    className="sm-panel-itemLabel"
                    ref={(el) => {
                      itemLabelsRef.current[idx] = el
                    }}
                  >
                    <VariableProximity
                      label={item.label}
                      containerRef={panelRef}
                      radius={80}
                      falloff="gaussian"
                      fromFontVariationSettings="'wght' 650, 'wdth' 95"
                      toFontVariationSettings="'wght' 900, 'wdth' 130"
                      className="sm-panel-itemText"
                    />
                  </span>
                  <span className="sm-panel-arrow" aria-hidden="true">
                    →
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  )
}

export default StaggeredMenu
