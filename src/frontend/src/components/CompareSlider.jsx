import { useState } from 'react'

export default function CompareSlider({ before, after }) {
  const [pos, setPos] = useState(50)
  const [beforeBroken, setBeforeBroken] = useState(false)

  return (
    <div className="compare">
      <div className="compare-frame">
        {before && !beforeBroken && (
          <img
            className="compare-img"
            src={before}
            alt="input"
            onError={() => setBeforeBroken(true)}
            draggable={false}
          />
        )}
        <div
          className="compare-clip"
          style={before && !beforeBroken ? { clipPath: `inset(0 0 0 ${pos}%)` } : undefined}
        >
          <img className="compare-img" src={after} alt="synthetic output" draggable={false} />
        </div>
        {before && !beforeBroken && (
          <>
            <div className="compare-handle" style={{ left: `${pos}%` }} />
            <span className="compare-tag left">input</span>
            <span className="compare-tag right">synthetic</span>
          </>
        )}
        <span className="watermark">SYNTHETIC</span>
      </div>

      {before && !beforeBroken && (
        <input
          className="compare-range"
          type="range"
          min="0"
          max="100"
          value={pos}
          onChange={(e) => setPos(Number(e.target.value))}
          aria-label="Compare input and synthetic output"
        />
      )}
      {beforeBroken && (
        <p className="hint">Input preview unavailable in-browser (whole-slide); showing output only.</p>
      )}
    </div>
  )
}
