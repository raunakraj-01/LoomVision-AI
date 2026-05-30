# Product Requirements Document
## LoomVision AI — 3D Frontend Experience

| Field | Details |
|---|---|
| **Project** | LoomVision AI |
| **Module** | 3D Frontend (Web Client) |
| **Version** | v1.0 |
| **Status** | Draft |
| **Author** | [Your Name] |
| **Last Updated** | May 2026 |

---

## 1. Executive Summary

LoomVision AI is an AI-powered visual intelligence platform. This PRD defines the requirements for its **3D interactive frontend** — a browser-based experience that communicates the product's depth and capability through immersive 3D rendering, real-time AI feedback, and cinematic micro-interactions.

The 3D frontend serves as the primary interface layer for users interacting with LoomVision's AI analysis pipeline. It must feel fast, intelligent, and visually differentiated — setting the product apart from flat-UI competitors and demonstrating engineering depth suitable for a placement portfolio or production deployment.

---

## 2. Problem Statement

### 2.1 User Problem
Users working with AI vision tools are overwhelmed by static, form-based interfaces that fail to represent the spatial and temporal nature of visual AI tasks (object detection, 3D scene reconstruction, video analysis, loom/textile pattern recognition, etc.).

### 2.2 Product Gap
Existing AI vision platforms provide 2D dashboards that:
- Do not convey spatial depth or model confidence visually
- Lack real-time feedback loops in the UI
- Feel generic and fail to build trust in the AI's capability

### 2.3 Opportunity
A 3D-first frontend positions LoomVision AI as a premium, differentiated product and allows the underlying AI results (depth maps, bounding volumes, skeletal overlays, pattern meshes) to be visualized natively — in the medium they were computed in.

---

## 3. Goals & Non-Goals

### Goals
- Build an immersive, performant 3D web interface using **Three.js / React Three Fiber**
- Visualize AI model outputs (bounding boxes, heatmaps, depth clouds, mesh overlays) in 3D space
- Deliver a sub-3s initial load time with lazy hydration of heavy 3D assets
- Achieve 60 FPS rendering on mid-range devices (M1 MacBook, mid-tier Android)
- Make the product placement-ready: clean architecture, documented codebase, Lighthouse score ≥ 90

### Non-Goals
- Native mobile app (out of scope for v1; web-responsive suffices)
- Real-time multiplayer or collaborative sessions
- Custom WebGPU renderer (Three.js WebGL pipeline is sufficient for v1)
- Backend AI model training UI

---

## 4. Target Users

| Persona | Description | Core Need |
|---|---|---|
| **ML Engineer** | Reviews model outputs and debug sessions | Accurate 3D overlay of inference results |
| **Product Manager** | Evaluates AI capabilities for stakeholder demos | Wow-factor, shareable demos |
| **End User (Vision App)** | Submits images/videos for analysis | Simple upload → rich 3D result |
| **Recruiter / Evaluator** | Reviews project for placement | Clean code, architecture doc, live demo |

---

## 5. User Stories

### 5.1 Core Flows

**US-01 — Landing & Onboarding**
> As a new user, I want to land on a visually striking 3D scene that communicates what LoomVision does within 5 seconds, so I understand the product without reading documentation.

**Acceptance Criteria:**
- Hero section renders an animated 3D scene (rotating AI mesh / loom pattern) on load
- Scene includes a CTA ("Try LoomVision" / "Upload Image")
- Particle system or environment map communicates "AI intelligence" aesthetic
- No layout shift (CLS < 0.1)

---

**US-02 — File Upload & Input Panel**
> As a user, I want to drag-and-drop an image or video into the interface so the AI can analyze it.

**Acceptance Criteria:**
- Drop zone accepts `.jpg`, `.png`, `.mp4`, `.webm`, `.obj`, `.glb`
- Upload progress shown via animated 3D progress ring (not a flat bar)
- File preview rendered as a 3D card/plane in the scene
- Error states (wrong format, size > 50MB) shown with animated toast

---

**US-03 — 3D AI Result Visualization**
> As a user, after upload I want to see AI-detected objects, patterns, or depth maps rendered in 3D space so I can understand the model's output spatially.

**Acceptance Criteria:**
- Detected bounding boxes rendered as semi-transparent 3D cuboids with label tags
- Depth map rendered as a point cloud or extruded height map
- Confidence score shown as opacity/glow intensity of each 3D object
- Camera auto-orbits to a good viewing angle, then user can take control (OrbitControls)
- Animations: objects fade in with stagger (0–800ms per object)
- Toggle panel: switch between Point Cloud / Mesh / Bounding Box views

---

**US-04 — Interactive 3D Scene Controls**
> As a user, I want to rotate, zoom, and pan the 3D result scene so I can inspect it from any angle.

**Acceptance Criteria:**
- Mouse drag → orbit; scroll → zoom; right-click drag → pan
- Touch support: pinch-to-zoom, two-finger pan
- "Reset Camera" button snaps back to default view (animated)
- Keyboard shortcuts: `R` = reset, `F` = fit-to-frame, `G` = toggle grid

---

**US-05 — Analysis Dashboard (Sidebar)**
> As a user, I want a sidebar panel showing detected labels, confidence scores, and metadata so I can cross-reference the 3D view with raw data.

**Acceptance Criteria:**
- Collapsible sidebar with a list of detected entities
- Hovering a sidebar item highlights the corresponding 3D object
- Clicking an item flies camera to that object (smooth tween)
- Export button: download results as JSON or PNG screenshot

---

**US-06 — Performance & Responsiveness**
> As a user on a laptop or tablet, I want the 3D experience to run smoothly without lag or jank.

**Acceptance Criteria:**
- Target: 60 FPS on desktop, ≥ 30 FPS on mid-tier mobile
- LOD (Level of Detail) switching for complex meshes beyond 50k vertices
- Canvas degrades gracefully: if WebGL unavailable, show 2D fallback with static image
- Bundle size: initial JS bundle ≤ 300 KB gzipped (Three.js chunk loaded async)

---

**US-07 — Dark / Light Theme**
> As a user, I want to toggle between dark and light mode so the interface matches my system preference or preference.

**Acceptance Criteria:**
- System preference detected on first load (`prefers-color-scheme`)
- Manual toggle in top nav
- 3D scene environment (background, lighting, fog) adapts to theme
- Transition: smooth 400ms CSS variable interpolation

---

## 6. Technical Architecture

### 6.1 Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| **Framework** | React 18 + Vite | Fast HMR, ecosystem maturity |
| **3D Engine** | React Three Fiber (R3F) + Three.js | Declarative 3D in React tree |
| **3D Helpers** | @react-three/drei | OrbitControls, loaders, helpers |
| **Physics** (optional v1.1) | @react-three/rapier | Cloth/soft-body simulation for loom patterns |
| **Animation** | GSAP + Framer Motion | Timeline control + UI transitions |
| **State Management** | Zustand | Lightweight, 3D-scene-friendly |
| **Styling** | Tailwind CSS + CSS Modules | Utility + scoped 3D overlay styles |
| **API Layer** | Axios + React Query | AI backend calls with caching |
| **Testing** | Vitest + React Testing Library | Unit + component tests |
| **Deployment** | Vercel / Netlify | Preview deploys per PR |

### 6.2 File Structure

```
loomvision-frontend/
├── public/
│   ├── models/            # Static .glb / .gltf assets
│   └── textures/          # Environment maps, HDR, matcaps
├── src/
│   ├── components/
│   │   ├── canvas/        # All R3F / Three.js components
│   │   │   ├── Scene.jsx           # Root R3F canvas
│   │   │   ├── BoundingBoxes.jsx   # AI result overlays
│   │   │   ├── PointCloud.jsx      # Depth map renderer
│   │   │   ├── LoomMesh.jsx        # Pattern mesh visualization
│   │   │   └── Environment.jsx     # Lighting, HDR, fog
│   │   ├── ui/            # 2D overlay components (Tailwind)
│   │   │   ├── Sidebar.jsx
│   │   │   ├── UploadZone.jsx
│   │   │   ├── ControlsHUD.jsx
│   │   │   └── Navbar.jsx
│   │   └── shared/        # Buttons, Toasts, Loaders
│   ├── hooks/
│   │   ├── useAIResults.js       # React Query fetcher for AI API
│   │   ├── useCameraControl.js   # Programmatic camera tweens
│   │   └── useTheme.js
│   ├── store/
│   │   └── sceneStore.js         # Zustand global state
│   ├── utils/
│   │   ├── geometryUtils.js      # Mesh processing helpers
│   │   └── colorMap.js           # Confidence → color mapping
│   ├── pages/
│   │   ├── Landing.jsx
│   │   ├── Workspace.jsx
│   │   └── Results.jsx
│   └── App.jsx
├── tests/
├── .env.example
├── vite.config.js
└── README.md
```

### 6.3 Data Flow

```
User Uploads File
      │
      ▼
UploadZone (UI) ──► API POST /analyze ──► AI Backend
      │                                       │
      │                                  Returns JSON:
      │                               { objects[], depth_map,
      │                                 confidence[], mesh_url }
      │                                       │
      ▼                                       ▼
sceneStore.setResults()          Scene.jsx re-renders with:
                                  - BoundingBoxes (objects[])
                                  - PointCloud (depth_map)
                                  - Labels + confidence glow
```

---

## 7. API Contract (Frontend → Backend)

### POST `/api/v1/analyze`
**Request:**
```json
{
  "file_url": "https://cdn.loomvision.ai/uploads/abc123.jpg",
  "analysis_mode": "full" // "depth" | "detect" | "pattern" | "full"
}
```

**Response:**
```json
{
  "job_id": "jb_abc123",
  "status": "completed",
  "objects": [
    {
      "id": "obj_1",
      "label": "Fabric Roll",
      "confidence": 0.94,
      "bbox_3d": { "x": 0.2, "y": 0.1, "z": 0.5, "w": 0.4, "h": 0.6, "d": 0.3 }
    }
  ],
  "depth_map_url": "https://cdn.loomvision.ai/results/abc123_depth.exr",
  "mesh_url": "https://cdn.loomvision.ai/results/abc123.glb",
  "metadata": { "processing_ms": 1240, "model_version": "lv-vision-v2" }
}
```

---

## 8. Design Specifications

### 8.1 Visual Language
- **Theme:** Dark-first, deep navy/obsidian backgrounds, electric teal + amber accents
- **Typography:** Display — `Clash Display` or `Syne`; Body — `DM Sans`
- **3D Aesthetic:** Wireframe + translucent surfaces; holographic bounding boxes; depth-of-field blur on background geometry
- **Motion Principles:** Ease-out enters, ease-in exits; 3D elements spring into place; no jarring cuts

### 8.2 Color Palette

| Token | Value | Use |
|---|---|---|
| `--bg-primary` | `#080D18` | Canvas background |
| `--bg-surface` | `#111827` | Cards, sidebar |
| `--accent-teal` | `#00D4AA` | Primary CTA, highlights |
| `--accent-amber` | `#FFB347` | Confidence > 90% |
| `--accent-red` | `#FF6B6B` | Confidence < 50%, errors |
| `--text-primary` | `#F0F4FF` | Headings |
| `--text-muted` | `#6B7A99` | Secondary labels |

### 8.3 Key Screens / Views

| Screen | Description |
|---|---|
| **Landing** | Hero 3D animation, value prop, CTA |
| **Workspace** | Full-canvas 3D scene + sidebar + upload zone |
| **Results** | Annotated 3D view + data panel + export |
| **Loading** | Animated 3D spinner (orbiting nodes) |
| **Error** | 3D broken mesh with friendly copy |

---

## 9. Performance Requirements

| Metric | Target |
|---|---|
| First Contentful Paint (FCP) | < 1.5s |
| Largest Contentful Paint (LCP) | < 2.5s |
| Time to Interactive (TTI) | < 3.5s |
| Cumulative Layout Shift (CLS) | < 0.1 |
| 3D Scene Frame Rate (desktop) | 60 FPS |
| 3D Scene Frame Rate (mobile) | ≥ 30 FPS |
| JS Bundle (initial, gzipped) | ≤ 300 KB |
| Lighthouse Score | ≥ 90 |

---

## 10. Accessibility

- All interactive 3D controls must have keyboard equivalents
- Screen-reader announcements for AI result summaries (via ARIA live region)
- Color not used as the sole indicator (shapes + labels always present)
- Reduced motion: honor `prefers-reduced-motion` — disable particle animations and auto-orbit
- Minimum contrast ratio: 4.5:1 for all text overlays on 3D canvas

---

## 11. Milestones & Timeline

| Phase | Scope | Duration |
|---|---|---|
| **Phase 1 — Foundation** | Project setup, Vite + R3F, routing, Zustand store, theme | Week 1 |
| **Phase 2 — Landing Page** | Hero 3D scene, animation, CTA, responsive layout | Week 2 |
| **Phase 3 — Upload + API** | Upload zone, API integration, loading states, error handling | Week 3 |
| **Phase 4 — 3D Results** | BoundingBoxes, PointCloud, LoomMesh, camera controls | Weeks 4–5 |
| **Phase 5 — Dashboard UI** | Sidebar, hover-to-highlight, camera fly-to, export | Week 6 |
| **Phase 6 — Polish & Perf** | LOD, bundle optimization, Lighthouse tuning, accessibility | Week 7 |
| **Phase 7 — Testing & Docs** | Vitest unit tests, Storybook, README, architecture diagram | Week 8 |

---

## 12. Success Metrics

| Metric | Definition | Target |
|---|---|---|
| **Demo Engagement** | Avg. time spent in 3D workspace | > 3 minutes |
| **Upload Completion Rate** | Users who upload and see results | > 75% |
| **Performance** | Lighthouse score across pages | ≥ 90 |
| **Placement Signal** | GitHub stars / recruiter mentions | — |
| **Error Rate** | JS errors per session | < 0.5% |

---

## 13. Open Questions

| # | Question | Owner | Status |
|---|---|---|---|
| 1 | Will the AI backend return `.glb` meshes or raw depth arrays? Affects renderer choice. | Backend Team | Open |
| 2 | Should loom pattern visualization support textile-specific UV overlays? | Design | Open |
| 3 | Is WebXR (AR/VR mode) in scope for v1.1? | PM | Deferred |
| 4 | Rate limits on `/analyze` endpoint — how to handle queuing in UI? | Backend | Open |

---

## 14. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Three.js bundle bloat | High | High | Dynamic import + code splitting per route |
| WebGL not available on older devices | Medium | Medium | 2D fallback page with static result images |
| AI API latency > 5s | Medium | High | Streaming progress endpoint + skeleton 3D placeholder |
| Complex mesh crashes browser tab | Low | High | Vertex budget cap (500k), LOD fallback |
| Scope creep on 3D features | Medium | Medium | Strict phase gate reviews before Phase 4 |

---

## 15. Appendix

### A. Reference Implementations
- [React Three Fiber Docs](https://docs.pmnd.rs/react-three-fiber)
- [Three.js Journey](https://threejs-journey.com) — Shader & performance patterns
- [Drei Helpers](https://github.com/pmndrs/drei) — OrbitControls, useGLTF, Environment
- [Leva](https://github.com/pmndrs/leva) — In-scene debug panel

### B. Competitor Reference UIs
- Runway ML — clean 2D but no 3D depth
- Roboflow — functional but not visually differentiated
- Spline — 3D first, but no AI result overlay

### C. Glossary

| Term | Definition |
|---|---|
| **R3F** | React Three Fiber — React renderer for Three.js |
| **GLTF / GLB** | Open 3D model format used for AI mesh output |
| **Point Cloud** | 3D visualization of depth map as a set of colored 3D points |
| **BBox 3D** | 3D bounding box output from object detection model |
| **LOD** | Level of Detail — rendering meshes at lower resolution at a distance |
| **EXR** | High-dynamic-range image format used for depth maps |
| **OrbitControls** | Three.js helper for mouse-driven camera rotation/zoom/pan |

---

*This PRD is a living document. Update it as requirements evolve and decisions are made.*
