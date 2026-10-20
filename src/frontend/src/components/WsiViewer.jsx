import { useEffect, useRef } from 'react'
import OpenSeadragon from 'openseadragon'

export default function WsiViewer({ src, onClose }) {
  const host = useRef(null)

  useEffect(() => {
    if (!host.current || !src) return undefined
    const viewer = OpenSeadragon({
      element: host.current,
      tileSources: { type: 'image', url: src },
      showNavigationControl: false,
      showNavigator: false,
      gestureSettingsMouse: { clickToZoom: true, dblClickToZoom: true },
      maxZoomPixelRatio: 8,
      visibilityRatio: 0.5,
    })
    return () => viewer.destroy()
  }, [src])

  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Whole slide viewer">
      <div className="modal-head">
        <h2>Whole-slide view</h2>
        <div>
          <span className="hint">scroll to zoom · drag to pan</span>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
      <div className="modal-canvas" ref={host} />
    </div>
  )
}
