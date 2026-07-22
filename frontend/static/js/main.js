/**
 * FaceWarp Lab — main.js
 * Handles: file upload, slider live-preview, AJAX processing,
 * Matplotlib FFT visualisation, metrics rendering, export, gallery, lightbox.
 */

'use strict';

/* ═══════════════════════════════════════════════
   State
═══════════════════════════════════════════════ */
const state = {
  file:             null,
  sessionId:        null,            // set after successful POST /api/upload
  imageId:          null,
  mode:             'expression',    // 'expression' | 'aging' | 'accessory' | 'ai_expression' | 'virtual_tryon'
  agingIntensity:   1.0,
  smileIntensity:   0.5,
  eyebrowHeight:    0.0,
  faceSlimming:     0.0,
  lipIntensity:     0.0,
  aiExpressionPreset: 'smile',
  aiExpressionIntensity: 0.65,
  aiUseDrivingTemplate: false,
  aiDrivingTemplate: 'direct_smile',
  aiCandidateFrameOverride: 'auto',
  accessoryItem:    'glasses',       // 'glasses' | 'hat' | 'makeup'
  assetManifest:    null,
  storeManifest:    null,
  palettes:         null,
  selectedAccessoryCategory: 'glasses',
  selectedAssetId:  'default_black',
  selectedAccessories: [],
  editorTab:        'accessories',
  autoApplyTimer:   null,
  autoApplyQueued:  false,
  lastAutoApplyReason: '-',
  lastEditorPayload: null,
  lastEffectsMeta:  null,
  editorColors: {
    hair: '#6F3BB8',
    eye: '#3F7FBF',
    lipstick: '#B00020',
    beard: '#1a1108',
    blush: '#E8A0A8',
    eyeshadow: '#8C7A6B',
    eyeliner: '#080808',
  },
  webcam: {
    active: false,
    stream: null,
    sessionId: null,
    inFlight: false,
    trackingInFlight: false,
    rafId: null,
    lastFrameAt: 0,
    lastTrackAt: 0,
    lastRawFrameAt: 0,
    lastFpsAt: 0,
    framesSinceFps: 0,
    targetFps: 30,
    trackingFps: 24,
    captureMaxSize: 224,
    effectMaxSize: 360,
    renderMaxSize: 420,
    rawMaxSize: 480,
    sourceCanvas: null,
    lastLandmarks: null,
    smoothedLandmarks: null,
    landmarkVelocity: null,
    lastSmoothAt: 0,
    lastLandmarkAt: 0,
    processedFrameImage: null,
    processedFrameObjectUrl: null,
    processedFrameInFlight: false,
    lastProcessedFrameAt: 0,
    backendFrameFps: 24,
    backendConsecutiveErrors: 0,
    browserEffectsReady: false,
    browserEffectsLoading: false,
    browserEffectsUnavailable: false,
    browserEffectsError: null,
    visionResolver: null,
    faceLandmarker: null,
    hairSegmenter: null,
    hairSegmenterAvailable: false,
    hairSegmentationInFlight: false,
    lastHairSegmentAt: 0,
    hairSegmentFps: 30,
    browserHairMask: null,
    browserHairMaskWidth: 0,
    browserHairMaskHeight: 0,
    browserHairMaskCoverage: 0,
    lastBrowserLandmarksAt: 0,
    browserFallbackBackend: false,
    assetImageCache: new Map(),
    lastObjectUrl: null,
  },
  tryon: {
    garmentFile: null,
    selectedStoreItemId: null,
    selectedStoreSlot: 'upperbody',
    modelType: 'dc',
    category: 'upperbody',
    steps: 20,
    scale: 2.0,
    sample: 1,
    seed: -1,
    status: null,
  },
  showLandmarks:    false,
  showGrayscale:    false,   
  agingAlgorithm:   'frequency', 
  processing:       false,
  resultData:       null,            // last API response
  // Cached grayscale data URLs (set after process, cleared on new upload)
  grayscaleOrigUrl: null,            // histogram-equalized grayscale of original
  grayscaleResUrl:  null,            // histogram-equalized grayscale of result (if returned)
};

let _suppressAutoApply = false;
let _editorAutoApplyDebounceTimer = null;

/* ═══════════════════════════════════════════════
   DOM refs
═══════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

const dom = {
  // Dropzone
  dropzone:            $('dropzone'),
  fileInput:           $('file-input'),
  dropzonePreview:     $('dropzone-preview'),
  previewThumb:        $('preview-thumb'),
  previewName:         $('preview-name'),
  dropzoneIcon:        $('dropzone-icon'),
  groupExpression:     $('group-expression'),

  // Mode buttons
  modeBtns:            document.querySelectorAll('.mode-btn'),
  modeBadge:           $('mode-badge'),
  algoFreq:            $('algo-freq'),
  algoAi:              $('algo-ai'),

  // Sliders
  sliderAging:         $('slider-aging'),
  sliderSmile:         $('slider-smile'),
  sliderEyebrow:       $('slider-eyebrow'),
  sliderSlim:          $('slider-slim'),
  sliderLip:           $('slider-lip'),
  valLip:              $('val-lip'),
  aiExpressionPreset:  $('ai-expression-preset'),
  aiExpressionIntensity: $('ai-expression-intensity'),
  aiExpressionIntensityValue: $('ai-expression-intensity-value'),
  aiUseDrivingTemplate: $('ai-use-driving-template'),
  aiDrivingTemplate: $('ai-driving-template'),
  aiCandidateFrameOverride: $('ai-candidate-frame-override'),

  // Slider value displays
  valAging:            $('val-aging'),
  valSmile:            $('val-smile'),
  valEyebrow:          $('val-eyebrow'),
  valSlim:             $('val-slim'),

  // Toggles
  landmarkToggle:      $('landmark-toggle'),
  grayscaleToggle:     $('grayscale-toggle'),

  // Buttons
  btnProcess:          $('btn-process'),
  btnAiExpression:     $('btn-ai-expression'),
  btnWebcam:           $('btn-webcam'),
  btnWebcamCapture:    $('btn-webcam-capture'),
  btnWebcamClose:      $('btn-webcam-close'),
  btnProcessLabel:     $('btn-process-label'),
  btnProcessSpin:      $('btn-process-spinner'),
  btnExport:           $('btn-export'),
  btnAnalytics:        $('btn-analytics'),

  // Image stages
  imgOriginal:         $('img-original'),
  imgResult:           $('img-result'),
  imgGrayscaleOrig:    $('img-grayscale-original'),
  imgGrayscaleResult:  $('img-grayscale-result'),
  labelOriginal:       $('label-original'),
  labelResult:         $('label-result'),
  resultSpinner:       $('result-spinner'),
  resultPlaceholder:   $('result-placeholder'),
  processingBadge:     $('processing-badge'),
  grayscaleBadge:      $('grayscale-badge'),
  canvasLmOrig:        $('canvas-landmarks-original'),
  canvasLmResult:      $('canvas-landmarks-result'),
  mainStageRow:        $('main-stage-row'),
  webcamStage:         $('webcam-stage-container'),
  webcamVideo:         $('webcam-video'),
  webcamRawCanvas:     $('webcam-raw-canvas'),
  webcamCanvas:        $('webcam-canvas'),
  webcamSpinner:       $('webcam-spinner'),
  webcamFpsBadge:      $('webcam-fps-badge'),

  // Analytics drawer
  analyticsDrawer:     $('analytics-drawer'),
  analyticsChevron:    $('analytics-chevron'),

  // FFT Images (Matplotlib)
  imgFftOrig:          $('img-fft-original'),
  imgFftResult:        $('img-fft-result'),
  fftOrigPholder:      $('fft-original-placeholder'),
  fftResultPholder:    $('fft-result-placeholder'),
  imgFftPhaseOrig:     $('img-fft-phase-original'),
  imgFftPhaseResult:   $('img-fft-phase-result'),
  fftPhaseOrigPholder: $('fft-phase-original-placeholder'),
  fftPhaseResultPholder: $('fft-phase-result-placeholder'),

  // Metrics (strip)
  metricsStrip:        $('metrics-strip'),
  metricMse:           $('metric-mse'),
  metricPsnr:          $('metric-psnr'),
  metricSsim:          $('metric-ssim'),

  // Metrics table
  tblMse:              $('tbl-mse'),
  tblPsnr:             $('tbl-psnr'),
  tblSsim:             $('tbl-ssim'),
  tblTime:             $('tbl-time'),
  tblMseRating:        $('tbl-mse-rating'),
  tblPsnrRating:       $('tbl-psnr-rating'),
  tblSsimRating:       $('tbl-ssim-rating'),

  // Param snapshot
  snapMode:            $('snap-mode'),
  snapAging:           $('snap-aging'),
  snapSmile:           $('snap-smile'),
  snapEyebrow:         $('snap-eyebrow'),
  snapSlim:            $('snap-slim'),
  snapLandmarks:       $('snap-landmarks'),
  snapGrayscale:       $('snap-grayscale'),

  // Status
  statusDot:           $('status-dot'),
  statusText:          $('status-text'),

  // Toast
  toast:               $('toast'),
  toastIcon:           $('toast-icon'),
  toastMsg:            $('toast-msg'),

  // Group aging (hidden in expression mode)
  groupAging:          $('group-aging'),
  lblSliderAging:      $('lbl-slider-aging'),
  labelsSliderAging:   $('labels-slider-aging'),

  // Group accessory (hidden unless accessory mode)
  groupAccessory:      $('group-accessory'),
  accBtns:             document.querySelectorAll('.acc-btn'),

  // Virtual try-on
  groupTryon:          $('group-tryon'),
  tryonStoreOpen:      $('tryon-store-open'),
  tryonStoreClose:     $('tryon-store-close'),
  tryonStoreModal:     $('tryon-store-modal'),
  tryonStoreTabs:      $('tryon-store-tabs'),
  tryonStoreGrid:      $('tryon-store-grid'),
  tryonStoreSelected:  $('tryon-store-selected'),
  tryonStoreSelectedThumb: $('tryon-store-selected-thumb'),
  tryonStoreSelectedEmpty: $('tryon-store-selected-empty'),
  tryonStoreSelectedName: $('tryon-store-selected-name'),
  tryonStoreSelectedMeta: $('tryon-store-selected-meta'),
  tryonStoreClear:     $('tryon-store-clear'),
  tryonStoreCount:     $('tryon-store-count'),
  tryonGarmentInput:   $('tryon-garment-input'),
  tryonGarmentName:    $('tryon-garment-name'),
  tryonModelType:      $('tryon-model-type'),
  tryonCategory:       $('tryon-category'),
  tryonSteps:          $('tryon-steps'),
  tryonStepsValue:     $('tryon-steps-value'),
  tryonScale:          $('tryon-scale'),
  tryonScaleValue:     $('tryon-scale-value'),
  tryonStatusNote:     $('tryon-status-note'),

  // Grayscale note
  grayscaleNote:       $('grayscale-note'),

  // Experimental AI expression debug
  aiExpressionPanel:   $('ai-expression-dev-panel'),
  aiExpressionDebug:   $('ai-expression-debug'),
  aiDebugProvider:     $('ai-debug-provider'),
  aiDebugMode:         $('ai-debug-mode'),
  aiDebugTemplate:     $('ai-debug-template'),
  aiDebugPreset:       $('ai-debug-preset'),
  aiDebugFrameCount:   $('ai-debug-frame-count'),
  aiDebugSelectedFrame: $('ai-debug-selected-frame'),
  aiDebugExpressionScore: $('ai-debug-expression-score'),
  aiDebugTopFrames:    $('ai-debug-top-frames'),
  aiDebugScoring:      $('ai-debug-scoring'),
  aiDebugCandidateDir: $('ai-debug-candidate-dir'),
  aiDebugBrowPx:       $('ai-debug-brow-px'),
  aiDebugEyePx:        $('ai-debug-eye-px'),
  aiDebugIrisPx:       $('ai-debug-iris-px'),
  aiDebugLiftPx:       $('ai-debug-lift-px'),
  aiDebugFiles:        $('ai-debug-files'),
  aiDebugBridge:       $('ai-debug-bridge'),
  aiDebugFallback:     $('ai-debug-fallback'),
  aiDebugError:        $('ai-debug-error'),
  aiExpressionFallbackNote: $('ai-expression-fallback-note'),
};

/* ═══════════════════════════════════════════════
   Toast helper
═══════════════════════════════════════════════ */
let toastTimer = null;
function showToast(msg, type = 'info', duration = 3500) {
  const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
  dom.toastIcon.textContent = icons[type] ?? 'ℹ️';
  dom.toastMsg.textContent  = msg;
  dom.toast.classList.remove('hidden');
  dom.toast.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    dom.toast.classList.remove('visible');
    setTimeout(() => dom.toast.classList.add('hidden'), 350);
  }, duration);
}

/* ═══════════════════════════════════════════════
   Status bar
═══════════════════════════════════════════════ */
function setStatus(text, type = 'idle') {
  dom.statusText.textContent = text;
  dom.statusDot.className = 'w-2 h-2 rounded-full animate-pulse-slow';
  const colors = {
    idle:       'bg-slate-600',
    processing: 'bg-brand-500',
    success:    'bg-emerald-500',
    error:      'bg-red-500',
  };
  dom.statusDot.classList.add(colors[type] ?? colors.idle);
}

/* ═══════════════════════════════════════════════
   Slider updates
═══════════════════════════════════════════════ */
function fmt(v, decimals = 2) {
  return parseFloat(v).toFixed(decimals);
}

function bindSlider(slider, display, stateKey) {
  if (!slider || !display) return;
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    const step = parseFloat(slider.step || 1);
    const decimals = step < 1 ? 2 : 0;
    display.textContent = fmt(v, decimals);
    state[stateKey] = v;
    updateParamSnapshot();
  });
}

bindSlider(dom.sliderAging,   dom.valAging,   'agingIntensity');
bindSlider(dom.sliderSmile,   dom.valSmile,   'smileIntensity');
bindSlider(dom.sliderEyebrow, dom.valEyebrow, 'eyebrowHeight');
bindSlider(dom.sliderLip,     dom.valLip,     'lipIntensity');
bindSlider(dom.sliderSlim,    dom.valSlim,    'faceSlimming');
bindSlider(dom.aiExpressionIntensity, dom.aiExpressionIntensityValue, 'aiExpressionIntensity');

if (dom.aiExpressionPreset) {
  dom.aiExpressionPreset.value = 'smile';
  dom.aiExpressionPreset.addEventListener('change', () => {
    state.aiExpressionPreset = dom.aiExpressionPreset.value || 'smile';
    const template = aiTemplateForPreset(state.aiExpressionPreset);
    state.aiDrivingTemplate = template;
    state.aiUseDrivingTemplate = template !== 'direct_smile';
    if (dom.aiDrivingTemplate) dom.aiDrivingTemplate.value = template;
    if (dom.aiUseDrivingTemplate) dom.aiUseDrivingTemplate.checked = state.aiUseDrivingTemplate;
  });
}

if (dom.aiUseDrivingTemplate) {
  dom.aiUseDrivingTemplate.addEventListener('change', () => {
    state.aiUseDrivingTemplate = dom.aiUseDrivingTemplate.checked;
    if (!state.aiUseDrivingTemplate) {
      state.aiDrivingTemplate = 'direct_smile';
      if (dom.aiDrivingTemplate) dom.aiDrivingTemplate.value = 'direct_smile';
    }
  });
}

if (dom.aiDrivingTemplate) {
  dom.aiDrivingTemplate.value = 'direct_smile';
  dom.aiDrivingTemplate.addEventListener('change', () => {
    state.aiDrivingTemplate = dom.aiDrivingTemplate.value || 'direct_smile';
    state.aiUseDrivingTemplate = state.aiDrivingTemplate !== 'direct_smile';
    if (dom.aiUseDrivingTemplate) {
      dom.aiUseDrivingTemplate.checked = state.aiUseDrivingTemplate;
    }
  });
}

if (dom.aiCandidateFrameOverride) {
  dom.aiCandidateFrameOverride.value = 'auto';
  dom.aiCandidateFrameOverride.addEventListener('change', () => {
    state.aiCandidateFrameOverride = dom.aiCandidateFrameOverride.value || 'auto';
  });
}

function aiInternalPresetFor(template, useDrivingTemplate) {
  if (!useDrivingTemplate) return 'natural_smile';
  const presetMap = {
    laugh: 'laugh',
    open_lip: 'open_lip',
    wink: 'wink',
    shy: 'natural_smile',
    aggrieved: 'aggrieved',
  };
  return presetMap[template] || 'natural_smile';
}

function aiTemplateForPreset(preset) {
  const presetMap = {
    smile: 'direct_smile',
    eyebrow_raise: 'direct_smile',
    laugh: 'laugh',
    surprise: 'aggrieved',
    neutral: 'direct_smile',
    wink: 'wink',
    sad: 'aggrieved',
    angry: 'open_lip',
  };
  return presetMap[preset] || 'direct_smile';
}

function aiScoringPresetFor(preset, template, useDrivingTemplate) {
  if (!useDrivingTemplate) return preset === 'smile' ? 'natural_smile' : preset;
  if (preset === 'surprise') return 'aggrieved';
  if (preset === 'neutral') return 'neutral';
  if (preset === 'sad') return 'aggrieved';
  if (preset === 'angry') return 'surprise';
  if (preset === 'smile' && template === 'laugh') return 'natural_smile';
  return preset || aiInternalPresetFor(template, useDrivingTemplate);
}

async function loadEditorAssets() {
  try {
    const manifestCacheBust = Date.now();
    const [manifestRes, paletteRes, storeRes] = await Promise.all([
      fetch(`/api/assets/manifest?t=${manifestCacheBust}`),
      fetch('/api/assets/palettes'),
      fetch(`/api/store/manifest?t=${manifestCacheBust}`),
    ]);

    if (manifestRes.ok) state.assetManifest = await manifestRes.json();
    if (paletteRes.ok) state.palettes = await paletteRes.json();
    if (storeRes.ok) state.storeManifest = await storeRes.json();

    renderEditorPalettes();
    renderAssetPicker();
    renderStorePicker();
    renderAccessory3dControls();
    updateEditorDebug();
  } catch (err) {
    console.warn('[FaceWarp] Asset/palette load failed:', err);
  }
}

function makeSwatch(color, label, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.title = label || color;
  button.className = 'h-6 rounded border border-surface-500 transition-transform hover:scale-105';
  button.style.background = color;
  button.addEventListener('click', onClick);
  return button;
}

function setColorInput(id, value) {
  const input = $(id);
  if (!input) return;
  _suppressAutoApply = true;
  try {
    input.value = value;
  } finally {
    _suppressAutoApply = false;
  }
}

function renderSwatches(containerId, items, selectedHex, onSelect) {
  const container = $(containerId);
  if (!container) return;
  container.innerHTML = '';
  (items || []).forEach(item => {
    const swatch = makeSwatch(item.hex, item.label, () => onSelect(item));
    if (String(item.hex).toLowerCase() === String(selectedHex).toLowerCase()) {
      swatch.classList.add('ring-2', 'ring-purple-300', 'ring-offset-1', 'ring-offset-surface-800');
    }
    container.appendChild(swatch);
  });
}

function renderEditorPalettes() {
  const palettes = state.palettes || {};
  renderSwatches('editor-hair-swatches', palettes.hair_colors, state.editorColors.hair, item => {
    state.editorColors.hair = item.hex;
    setColorInput('editor-hair-color', item.hex);
    setChecked('editor-hair-enabled', true);
    renderEditorPalettes();
    updateEditorDebug();
    scheduleAutoApply('hair color swatch');
  });
  renderSwatches('editor-eye-swatches', palettes.eye_colors, state.editorColors.eye, item => {
    state.editorColors.eye = item.hex;
    setColorInput('editor-eye-color', item.hex);
    setChecked('editor-eye-enabled', true);
    renderEditorPalettes();
    updateEditorDebug();
    scheduleAutoApply('eye color swatch');
  });

  const makeup = palettes.makeup_colors || {};
  ['lipstick', 'beard', 'blush', 'eyeshadow', 'eyeliner'].forEach(kind => {
    renderSwatches(`editor-${kind}-swatches`, makeup[kind], state.editorColors[kind], item => {
      state.editorColors[kind] = item.hex;
      setColorInput(`editor-${kind}-color`, item.hex);
      setChecked(`editor-${kind}-enabled`, true);
      renderEditorPalettes();
      updateEditorDebug();
      scheduleAutoApply(`${kind} swatch`);
    });
  });
}

function setChecked(id, checked) {
  const input = $(id);
  if (input) input.checked = checked;
}

function renderAssetPicker() {
  const grid = $('editor-asset-grid');
  if (!grid) return;
  const category = state.selectedAccessoryCategory || 'glasses';
  const assets = state.assetManifest?.categories?.[category] || [];
  grid.innerHTML = '';

  if (!assets.length) {
    grid.innerHTML = '<p class="rounded border border-surface-600 px-2 py-1.5 text-[10px] text-slate-500">No local assets in this category.</p>';
    state.selectedAssetId = null;
    return;
  }

  assets.forEach(asset => {
    const isDisabledExperimentalHat = category === 'hats' && asset.category === 'baseball_cap';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'flex items-center justify-between gap-2 rounded border border-surface-600 px-2 py-1.5 text-left text-[10px] text-slate-300 hover:border-purple-400';
    if (isDisabledExperimentalHat) {
      button.disabled = true;
      button.classList.add('cursor-not-allowed', 'opacity-45', 'hover:border-surface-600');
    }
    if (asset.id === state.selectedAssetId) {
      button.classList.add('border-purple-400', 'bg-purple-500/10');
    }
    const imgSrc = asset.path ? `/${asset.path}` : '';
    button.innerHTML = `
      <span class="flex min-w-0 items-center gap-2">
        ${imgSrc ? `<img src="${imgSrc}" alt="" class="h-8 w-12 rounded bg-surface-900 object-contain" loading="lazy" />` : ''}
        <span class="font-medium text-[11px]">${asset.label || asset.name || asset.id}</span>
      </span>
      <span class="shrink-0 text-[9px] text-slate-500">${isDisabledExperimentalHat ? 'experimental' : (asset.placeholder ? 'placeholder' : '')}</span>
    `;
    button.addEventListener('click', () => {
      if (isDisabledExperimentalHat) return;
      state.selectedAssetId = asset.id;
      renderAccessory3dControls();
      upsertCurrentAccessory();
      renderAssetPicker();
      renderSelectedAccessories();
      updateEditorDebug();
      scheduleAutoApply('accessory asset selected');
    });
    grid.appendChild(button);
  });
}

function accessoryControlsValue(id, fallback) {
  const input = $(id);
  return input ? Number(input.value) : fallback;
}

function selectedAccessoryAsset() {
  const assets = state.assetManifest?.categories?.[state.selectedAccessoryCategory] || [];
  return assets.find(asset => asset.id === state.selectedAssetId) || null;
}

function setAccessoryRenderModeForCategory() {
  const modeSelect = $('editor-accessory-render-mode');
  if (!modeSelect) return;

  const asset = selectedAccessoryAsset();
  let modes = Array.isArray(asset?.render_modes) ? [...asset.render_modes] : [];
  if (asset?.asset_role === 'procedural_reference' && !asset.path) {
    modes = modes.filter(mode => mode !== 'overlay_2d');
  }
  let desired = 'overlay_2d';

  if (state.selectedAccessoryCategory === 'necklaces') desired = 'physics_3d';
  if (state.selectedAccessoryCategory === 'hats') desired = 'hat_light_inpaint';
  if (modes.length && !modes.includes(desired)) desired = modes[0];

  [...modeSelect.options].forEach(option => {
    option.disabled = modes.length ? !modes.includes(option.value) : false;
  });
  modeSelect.value = desired;
}

function renderAccessory3dControls() {
  const necklace = $('editor-necklace-3d-controls');
  const hat = $('editor-hat-3d-controls');
  const earringMotion = $('editor-earring-motion-controls');
  const isNecklace = state.selectedAccessoryCategory === 'necklaces';
  const isHat = state.selectedAccessoryCategory === 'hats';
  const isEarring = state.selectedAccessoryCategory === 'earrings';

  if (necklace) necklace.classList.toggle('hidden', !isNecklace);
  if (hat) hat.classList.toggle('hidden', !isHat);
  if (earringMotion) earringMotion.classList.toggle('hidden', !isEarring);
  setAccessoryRenderModeForCategory();
}

function accessoryMetadataForCurrentSelection(asset) {
  const defaults = (asset && typeof asset.default_metadata === 'object' && asset.default_metadata)
    ? { ...asset.default_metadata }
    : {};

  if (state.selectedAccessoryCategory === 'necklaces') {
    return {
      chain_length: accessoryControlsValue('editor-accessory-chain-length', defaults.chain_length ?? 1.0),
      chain_thickness: defaults.chain_thickness ?? 2.0,
      stiffness: accessoryControlsValue('editor-accessory-chain-stiffness', defaults.stiffness ?? 0.75),
      pendant_enabled: defaults.pendant_enabled ?? true,
      pendant_size: accessoryControlsValue('editor-accessory-pendant-size', defaults.pendant_size ?? 0.12),
      pendant_weight: defaults.pendant_weight ?? 1.0,
      material: $('editor-accessory-necklace-material')?.value || defaults.material || 'gold',
      anchor_mode: defaults.anchor_mode || 'clavicle_drape',
      contact_shadow: defaults.contact_shadow ?? true,
      metal_highlight: defaults.metal_highlight ?? 0.75,
      pearl_highlight: defaults.pearl_highlight ?? 0.65,
    };
  }

  if (state.selectedAccessoryCategory === 'hats') {
    return {
      color: $('editor-accessory-hat-color')?.value || defaults.color || '#222222',
      skull_fit: accessoryControlsValue('editor-accessory-hat-fit', defaults.skull_fit ?? 1.05),
      fold_height: accessoryControlsValue('editor-accessory-fold-height', defaults.fold_height ?? 0.18),
      top_sag: accessoryControlsValue('editor-accessory-top-sag', defaults.top_sag ?? 0.08),
      thickness: defaults.thickness ?? 0.08,
      material: defaults.material || 'fabric',
      fabric_texture_strength: defaults.fabric_texture_strength ?? 0.35,
      edge_softness: defaults.edge_softness ?? 0.65,
      contact_shadow: defaults.contact_shadow ?? true,
    };
  }

  if (state.selectedAccessoryCategory === 'earrings') {
    const explicitType = $('editor-accessory-earring-type')?.value || defaults.earring_type || 'auto';
    return {
      ...defaults,
      earring_type: explicitType === 'auto' ? (defaults.earring_type || 'auto') : explicitType,
      motion_preset: $('editor-accessory-earring-preset')?.value || defaults.motion_preset || 'normal',
      swing_intensity: accessoryControlsValue('editor-accessory-earring-swing', defaults.swing_intensity ?? 0.8),
    };
  }

  return defaults;
}

function addSelectedAccessory() {
  if (!state.selectedAssetId) {
    showToast('No asset selected in this category.', 'warning');
    return;
  }

  upsertCurrentAccessory();

  renderSelectedAccessories();
  updateEditorDebug();
  scheduleAutoApply('accessory added');
}

function currentAccessoryItem() {
  if (!state.selectedAssetId) return null;
  const asset = selectedAccessoryAsset();
  const renderMode = $('editor-accessory-render-mode')?.value || 'overlay_2d';
  const type = asset?.type || (state.selectedAccessoryCategory === 'hats' ? 'hat' : state.selectedAccessoryCategory);
  const category = asset?.category || state.selectedAccessoryCategory;
  const metadata = accessoryMetadataForCurrentSelection(asset);

  return {
    type,
    category,
    asset_id: state.selectedAssetId,
    render_mode: renderMode,
    metadata,
    scale: accessoryControlsValue('editor-accessory-scale', 1.0),
    offset_x: accessoryControlsValue('editor-accessory-offset-x', 0.0),
    offset_y: accessoryControlsValue('editor-accessory-offset-y', 0.0),
  };
}

function upsertCurrentAccessory() {
  const item = currentAccessoryItem();
  if (!item) return;

  const index = state.selectedAccessories.findIndex(existing =>
    existing.category === item.category
  );

  if (index >= 0) {
    state.selectedAccessories[index] = item;
  } else {
    state.selectedAccessories.push(item);
  }
}

function syncCategoryUIControls() {
  const category = state.selectedAccessoryCategory || 'glasses';
  const existing = state.selectedAccessories.find(item => item.category === category);
  
  if (existing) {
    state.selectedAssetId = existing.asset_id;
    
    const scaleInput = $('editor-accessory-scale');
    if (scaleInput) scaleInput.value = existing.scale;
    
    const offsetXInput = $('editor-accessory-offset-x');
    if (offsetXInput) offsetXInput.value = existing.offset_x;
    
    const offsetYInput = $('editor-accessory-offset-y');
    if (offsetYInput) offsetYInput.value = existing.offset_y;
    
    const renderModeInput = $('editor-accessory-render-mode');
    if (renderModeInput) renderModeInput.value = existing.render_mode;
    
    if (category === 'necklaces' && existing.metadata) {
      const material = $('editor-accessory-necklace-material');
      if (material && existing.metadata.material) material.value = existing.metadata.material;
      
      const chainLength = $('editor-accessory-chain-length');
      if (chainLength && existing.metadata.chain_length !== undefined) chainLength.value = existing.metadata.chain_length;
      
      const pendantSize = $('editor-accessory-pendant-size');
      if (pendantSize && existing.metadata.pendant_size !== undefined) pendantSize.value = existing.metadata.pendant_size;
      
      const stiffness = $('editor-accessory-chain-stiffness');
      if (stiffness && existing.metadata.stiffness !== undefined) stiffness.value = existing.metadata.stiffness;
    }
    
    if (category === 'hats' && existing.metadata) {
      const hatColor = $('editor-accessory-hat-color');
      if (hatColor && existing.metadata.color) hatColor.value = existing.metadata.color;
      
      const hatFit = $('editor-accessory-hat-fit');
      if (hatFit && existing.metadata.skull_fit !== undefined) hatFit.value = existing.metadata.skull_fit;
      
      const foldHeight = $('editor-accessory-fold-height');
      if (foldHeight && existing.metadata.fold_height !== undefined) foldHeight.value = existing.metadata.fold_height;
      
      const topSag = $('editor-accessory-top-sag');
      if (topSag && existing.metadata.top_sag !== undefined) topSag.value = existing.metadata.top_sag;
    }

    if (category === 'earrings' && existing.metadata) {
      const earringType = $('editor-accessory-earring-type');
      if (earringType && existing.metadata.earring_type) earringType.value = existing.metadata.earring_type;

      const preset = $('editor-accessory-earring-preset');
      if (preset && existing.metadata.motion_preset) preset.value = existing.metadata.motion_preset;

      const swing = $('editor-accessory-earring-swing');
      if (swing && existing.metadata.swing_intensity !== undefined) swing.value = existing.metadata.swing_intensity;
    }
  } else {
    const assets = state.assetManifest?.categories?.[category] || [];
    state.selectedAssetId = assets[0]?.id || null;
    
    const scaleInput = $('editor-accessory-scale');
    if (scaleInput) scaleInput.value = 1.0;
    
    const offsetXInput = $('editor-accessory-offset-x');
    if (offsetXInput) offsetXInput.value = 0;
    
    const offsetYInput = $('editor-accessory-offset-y');
    if (offsetYInput) offsetYInput.value = 0;
    
    let desiredRenderMode = 'overlay_2d';
    if (category === 'necklaces') desiredRenderMode = 'physics_3d';
    if (category === 'hats') desiredRenderMode = 'hat_light_inpaint';
    
    const renderModeInput = $('editor-accessory-render-mode');
    if (renderModeInput) renderModeInput.value = desiredRenderMode;
    
    if (category === 'necklaces') {
      const material = $('editor-accessory-necklace-material');
      if (material) material.value = 'gold';
      
      const chainLength = $('editor-accessory-chain-length');
      if (chainLength) chainLength.value = 1.0;
      
      const pendantSize = $('editor-accessory-pendant-size');
      if (pendantSize) pendantSize.value = 0.12;
      
      const stiffness = $('editor-accessory-chain-stiffness');
      if (stiffness) stiffness.value = 0.75;
    }
    
    if (category === 'hats') {
      const hatColor = $('editor-accessory-hat-color');
      if (hatColor) hatColor.value = '#222222';
      
      const hatFit = $('editor-accessory-hat-fit');
      if (hatFit) hatFit.value = 1.05;
      
      const foldHeight = $('editor-accessory-fold-height');
      if (foldHeight) foldHeight.value = 0.18;
      
      const topSag = $('editor-accessory-top-sag');
      if (topSag) topSag.value = 0.08;
    }

    if (category === 'earrings') {
      const earringType = $('editor-accessory-earring-type');
      if (earringType) earringType.value = 'auto';

      const preset = $('editor-accessory-earring-preset');
      if (preset) preset.value = 'normal';

      const swing = $('editor-accessory-earring-swing');
      if (swing) swing.value = 0.8;
    }
  }
}

function clearAccessoryAndMakeupState() {
  state.selectedAccessories = [];
  state.selectedAssetId = null;

  const checkboxIds = [
    'editor-hair-enabled',
    'editor-eye-enabled',
    'editor-skin-smooth-enabled',
    'editor-lipstick-enabled',
    'editor-beard-enabled',
    'editor-blush-enabled',
    'editor-eyeshadow-enabled',
    'editor-eyeliner-enabled'
  ];
  checkboxIds.forEach(id => {
    const el = $(id);
    if (el) el.checked = false;
  });

  renderEditorPalettes();
  renderAssetPicker();
  renderSelectedAccessories();
  updateEditorDebug();
}


function renderSelectedAccessories() {
  const list = $('editor-selected-accessories');
  if (!list) return;
  list.innerHTML = '';

  if (!state.selectedAccessories.length) {
    list.innerHTML = '<p>No accessories selected.</p>';
    return;
  }

  state.selectedAccessories.forEach((item, idx) => {
    const row = document.createElement('div');
    row.className = 'flex items-center justify-between rounded border border-surface-600 px-2 py-1';
    row.innerHTML = `<span>${item.type}:${item.asset_id} <em class="text-slate-500">${item.render_mode || 'overlay_2d'}</em></span><button type="button" class="text-red-300 hover:text-red-200">remove</button>`;
    row.querySelector('button').addEventListener('click', () => {
      state.selectedAccessories.splice(idx, 1);
      renderSelectedAccessories();
      updateEditorDebug();
      scheduleAutoApply('accessory removed');
    });
    list.appendChild(row);
  });
}

function editorEnabled(id) {
  const input = $(id);
  return Boolean(input?.checked);
}

function editorNumber(id, fallback) {
  const input = $(id);
  return input ? Number(input.value) : fallback;
}

function buildEditorEffectsPayload() {
  const effects = {
    hair_color: {
      enabled: editorEnabled('editor-hair-enabled'),
      color: $('editor-hair-color')?.value || state.editorColors.hair,
      intensity: editorNumber('editor-hair-intensity', 0.65),
    },
    eye_color: {
      enabled: editorEnabled('editor-eye-enabled'),
      color: $('editor-eye-color')?.value || state.editorColors.eye,
      intensity: editorNumber('editor-eye-intensity', 0.45),
    },
    makeup: {
      skin_smooth: {
        enabled: editorEnabled('editor-skin-smooth-enabled'),
        intensity: editorNumber('editor-skin-smooth-intensity', 0.25),
      },
      lipstick: {
        enabled: editorEnabled('editor-lipstick-enabled'),
        color: $('editor-lipstick-color')?.value || state.editorColors.lipstick,
        intensity: editorNumber('editor-lipstick-intensity', 0.7),
      },
      beard: {
        enabled: editorEnabled('editor-beard-enabled'),
        color: $('editor-beard-color')?.value || state.editorColors.beard,
        intensity: editorNumber('editor-beard-intensity', 0.6),
      },
      blush: {
        enabled: editorEnabled('editor-blush-enabled'),
        color: $('editor-blush-color')?.value || state.editorColors.blush,
        intensity: editorNumber('editor-blush-intensity', 0.35),
      },
      eyeshadow: {
        enabled: editorEnabled('editor-eyeshadow-enabled'),
        color: $('editor-eyeshadow-color')?.value || state.editorColors.eyeshadow,
        intensity: editorNumber('editor-eyeshadow-intensity', 0.4),
      },
      eyeliner: {
        enabled: editorEnabled('editor-eyeliner-enabled'),
        color: $('editor-eyeliner-color')?.value || state.editorColors.eyeliner,
        intensity: editorNumber('editor-eyeliner-intensity', 0.5),
      },
    },
    accessories: {
      enabled: state.selectedAccessories.length > 0,
      items: state.selectedAccessories,
    },
  };

  return effects;
}

function buildRealtimeEffectParams() {
  const effects = buildEditorEffectsPayload();
  const smile = Number(state.smileIntensity) || 0;
  const eyebrow = Number(state.eyebrowHeight) || 0;
  const lip = Number(state.lipIntensity) || 0;
  const slim = Number(state.faceSlimming) || 0;
  return {
    expression: {
      enabled: Math.abs(smile) > 0.001 || Math.abs(eyebrow) > 0.001 || Math.abs(lip) > 0.001,
      smile_intensity: smile,
      eyebrow_intensity: eyebrow,
      lip_intensity: lip,
    },
    face_reshape: {
      enabled: Math.abs(slim) > 0.001,
      face_slimming: slim,
      lip_intensity: 0,
    },
    hair_color: effects.hair_color,
    eye_color: effects.eye_color,
    skin_smooth: effects.makeup.skin_smooth,
    lipstick: effects.makeup.lipstick,
    beard: effects.makeup.beard,
    blush: effects.makeup.blush,
    eyeshadow: effects.makeup.eyeshadow,
    eyeliner: effects.makeup.eyeliner,
    accessories: effects.accessories,
  };
}

function updateEditorDebug() {
  const pre = $('editor-debug-json');
  const reason = $('editor-debug-reason');
  const status = $('editor-debug-effects-status');
  const effects = buildEditorEffectsPayload();

  if (reason) reason.textContent = state.lastAutoApplyReason || '-';

  if (status) {
    const meta = state.lastEffectsMeta;
    status.textContent = Array.isArray(meta)
      ? meta.map(item => `${item.effect}:${item.applied ? 'applied' : 'skipped'}`).join(', ')
      : '-';
  }

  if (!pre) return;
  pre.textContent = JSON.stringify(
    {
      effects,
      last_payload_effects: state.lastEditorPayload?.effects || null,
      last_effects_meta: state.lastEffectsMeta,
    },
    null,
    2,
  );
}

function renderEditorTab() {
  const makeupSection = $('editor-makeup-section');
  const accessorySection = $('editor-accessory-section');
  const showMakeup = state.editorTab === 'makeup';

  if (makeupSection) makeupSection.classList.toggle('hidden', !showMakeup);
  if (accessorySection) accessorySection.classList.toggle('hidden', showMakeup);
}

function runScheduledAutoApply(reason) {
  state.lastAutoApplyReason = reason || 'editor change';
  updateEditorDebug();

  if (state.webcam.active) {
    return;
  }

  if (!state.sessionId || !state.imageId) {
    return;
  }

  if (state.processing) {
    state.autoApplyQueued = true;
    return;
  }

  console.info(`[FaceWarp] auto apply: ${state.lastAutoApplyReason}`);
  clearTimeout(state.autoApplyTimer);
  setStatus(`Applying ${state.lastAutoApplyReason}...`, 'processing');
  state.autoApplyTimer = setTimeout(async () => {
    await processImage({
      autoReason: state.lastAutoApplyReason,
    });
  }, 350);
}

function scheduleAutoApply(reason) {
  if (_suppressAutoApply) return;
  clearTimeout(_editorAutoApplyDebounceTimer);
  _editorAutoApplyDebounceTimer = setTimeout(() => {
    runScheduledAutoApply(reason);
  }, 150);
}

function bindEditorControls() {
  const category = $('editor-accessory-category');
  if (category) {
    category.addEventListener('change', () => {
      state.selectedAccessoryCategory = category.value || 'glasses';
      syncCategoryUIControls();
      renderAccessory3dControls();
      renderAssetPicker();
      renderSelectedAccessories();
      updateEditorDebug();
    });
  }

  const addButton = $('editor-add-accessory');
  if (addButton) addButton.addEventListener('click', addSelectedAccessory);

  ['editor-hair-color', 'editor-eye-color'].forEach(id => {
    const input = $(id);
    if (input) {
      input.addEventListener('input', () => {
        if (_suppressAutoApply) return;
        if (id === 'editor-hair-color') state.editorColors.hair = input.value;
        if (id === 'editor-eye-color') state.editorColors.eye = input.value;
        if (id === 'editor-hair-color') setChecked('editor-hair-enabled', true);
        if (id === 'editor-eye-color') setChecked('editor-eye-enabled', true);
        updateEditorDebug();
        scheduleAutoApply(id === 'editor-hair-color' ? 'hair custom color' : 'eye custom color');
      });
    }
  });

  ['lipstick', 'beard', 'blush', 'eyeshadow', 'eyeliner'].forEach(kind => {
    const input = $(`editor-${kind}-color`);
    if (input) {
      input.addEventListener('input', () => {
        if (_suppressAutoApply) return;
        state.editorColors[kind] = input.value;
        setChecked(`editor-${kind}-enabled`, true);
        renderEditorPalettes();
        updateEditorDebug();
        scheduleAutoApply(`${kind} custom color`);
      });
    }
  });

  document.querySelectorAll('#group-accessory input, #group-accessory select').forEach(el => {
    el.addEventListener('input', updateEditorDebug);
    el.addEventListener('change', updateEditorDebug);
    el.addEventListener('input', () => {
      if (el.id === 'editor-accessory-render-mode') renderAccessory3dControls();
      if (el.id && el.id.startsWith('editor-accessory-')) upsertCurrentAccessory();
      scheduleAutoApply(el.id || 'editor input');
    });
    el.addEventListener('change', () => {
      if (el.id === 'editor-accessory-render-mode') renderAccessory3dControls();
      if (el.id && el.id.startsWith('editor-accessory-')) upsertCurrentAccessory();
      scheduleAutoApply(el.id || 'editor change');
    });
  });

  renderSelectedAccessories();
  renderAccessory3dControls();
  renderEditorTab();
}

function setTryonStatusNote(text, type = 'warning') {
  if (!dom.tryonStatusNote) return;
  dom.tryonStatusNote.textContent = text;
  const base = 'rounded-md p-2 text-[11px] leading-relaxed';
  const styles = {
    success: 'border border-emerald-400/20 bg-emerald-500/10 text-emerald-100',
    warning: 'border border-amber-400/20 bg-amber-500/10 text-amber-100',
    error: 'border border-red-400/20 bg-red-500/10 text-red-100',
    info: 'border border-brand-400/20 bg-brand-500/10 text-brand-100',
  };
  dom.tryonStatusNote.className = `${base} ${styles[type] || styles.info}`;
}

function tryonReasonLabel(reason) {
  const labels = {
    ootdiffusion_repo_missing: 'OOTDiffusion repo not found. Set FACEWARP_OOTDIFFUSION_ROOT or place it at D:\\2Testfile\\OOTDiffusion.',
    ootdiffusion_checkpoints_missing: 'OOTDiffusion checkpoints are missing. Add ootd, CLIP, OpenPose, and human parsing checkpoints under D:\\2Testfile\\OOTDiffusion\\checkpoints.',
    ootdiffusion_dependencies_missing: 'OOTDiffusion Python dependencies are missing in this environment.',
    ootdiffusion_requires_cuda: 'OOTDiffusion files are installed, but CUDA is not available. This try-on runtime requires an NVIDIA CUDA GPU; the current AMD/CPU environment cannot run it.',
  };
  return labels[reason] || reason || 'Virtual try-on runtime is not ready.';
}

async function refreshTryonStatus() {
  if (!dom.tryonStatusNote) return;
  try {
    const res = await fetch('/api/tryon/status');
    const data = await res.json().catch(() => ({}));
    const status = data.tryon || {};
    state.tryon.status = status;
    if (status.available) {
      setTryonStatusNote('OOTDiffusion runtime is ready.', 'success');
    } else if (status.preview_cpu_fallback_available) {
      setTryonStatusNote('CUDA try-on is unavailable; temporary CPU preview fallback is enabled for store/UI testing.', 'warning');
    } else if (status.installed && status.reason === 'ootdiffusion_requires_cuda') {
      setTryonStatusNote(tryonReasonLabel(status.reason), 'error');
    } else {
      setTryonStatusNote(tryonReasonLabel(status.reason), 'warning');
    }
  } catch (err) {
    state.tryon.status = null;
    setTryonStatusNote('Could not check OOTDiffusion runtime status.', 'error');
  }
}

function storeItemsForSelectedSlot() {
  const slot = state.tryon.selectedStoreSlot || 'upperbody';
  const items = state.storeManifest?.items || [];
  return items.filter(item => item.slot === slot);
}

const STORE_SLOT_LABELS = {
  upperbody: 'Upper Body',
  lowerbody: 'Lower Body',
  dress: 'Dresses',
  hat: 'Hats',
  glasses: 'Glasses',
  earrings: 'Earrings',
  necklace: 'Necklaces',
  hair_clip: 'Hair Clips',
};

function storeSlots() {
  const items = state.storeManifest?.items || [];
  const preferred = ['upperbody', 'lowerbody', 'dress'];
  const present = new Set(items.map(item => item.slot).filter(Boolean));
  return preferred.filter(slot => present.has(slot));
}

function selectedStoreItem() {
  const items = state.storeManifest?.items || [];
  return items.find(item => item.id === state.tryon.selectedStoreItemId) || null;
}

function setSelectedStoreLabel() {
  const item = selectedStoreItem();
  if (dom.tryonStoreSelected) {
    dom.tryonStoreSelected.textContent = item
      ? `${item.name || item.id} - ${item.pipeline || item.type}`
      : 'No store item selected.';
  }
  if (dom.tryonStoreSelectedName) {
    dom.tryonStoreSelectedName.textContent = item ? (item.name || item.id) : 'No cloth selected';
  }
  if (dom.tryonStoreSelectedMeta) {
    dom.tryonStoreSelectedMeta.textContent = item
      ? `${STORE_SLOT_LABELS[item.slot] || item.slot || 'Store'} - ${item.pipeline === 'virtual_tryon' ? 'virtual try-on' : 'accessory'}`
      : 'Choose a garment from the store.';
  }
  if (dom.tryonStoreSelectedThumb && dom.tryonStoreSelectedEmpty) {
    if (item?.thumbnail) {
      dom.tryonStoreSelectedThumb.src = `/${item.thumbnail}`;
      dom.tryonStoreSelectedThumb.classList.remove('hidden');
      dom.tryonStoreSelectedEmpty.classList.add('hidden');
    } else {
      dom.tryonStoreSelectedThumb.removeAttribute('src');
      dom.tryonStoreSelectedThumb.classList.add('hidden');
      dom.tryonStoreSelectedEmpty.classList.remove('hidden');
    }
  }
}

function openStoreModal() {
  if (!dom.tryonStoreModal) return;
  renderStorePicker();
  dom.tryonStoreModal.classList.remove('hidden');
  requestAnimationFrame(() => dom.tryonStoreModal.classList.remove('opacity-0'));
}

function closeStoreModal() {
  if (!dom.tryonStoreModal) return;
  dom.tryonStoreModal.classList.add('opacity-0');
  window.setTimeout(() => dom.tryonStoreModal?.classList.add('hidden'), 180);
}

function selectStoreItem(item) {
  if (!item.enabled) {
    showToast('This store item is not enabled or is missing fit assets.', 'warning');
    return;
  }
  state.tryon.selectedStoreItemId = item.id;
  state.tryon.garmentFile = null;
  if (dom.tryonGarmentInput) dom.tryonGarmentInput.value = '';
  if (dom.tryonGarmentName) dom.tryonGarmentName.textContent = 'No garment selected.';
  if (item.pipeline === 'virtual_tryon') {
    state.tryon.category = item.tryon_category || item.slot || 'upperbody';
    state.tryon.modelType = item.model_type || state.tryon.modelType || 'dc';
    if (dom.tryonCategory) dom.tryonCategory.value = state.tryon.category;
    if (dom.tryonModelType) dom.tryonModelType.value = state.tryon.modelType;
    setSelectedStoreLabel();
    renderStorePicker();
    closeStoreModal();
  } else if (item.pipeline === 'accessory_overlay') {
    applyStoreAccessoryItem(item);
    setSelectedStoreLabel();
    renderStorePicker();
    closeStoreModal();
  }
}

function renderStoreTabs() {
  if (!dom.tryonStoreTabs) return;
  const slots = storeSlots();
  dom.tryonStoreTabs.innerHTML = '';
  slots.forEach(slot => {
    const count = (state.storeManifest?.items || []).filter(item => item.slot === slot).length;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'rounded-md border px-3 py-2 text-xs font-semibold transition-colors';
    if (slot === state.tryon.selectedStoreSlot) {
      button.classList.add('border-brand-400', 'bg-brand-500/20', 'text-brand-100');
    } else {
      button.classList.add('border-surface-600', 'bg-surface-800', 'text-slate-400', 'hover:border-brand-400', 'hover:text-white');
    }
    button.textContent = `${STORE_SLOT_LABELS[slot] || slot} (${count})`;
    button.addEventListener('click', () => {
      state.tryon.selectedStoreSlot = slot;
      renderStorePicker();
    });
    dom.tryonStoreTabs.appendChild(button);
  });
}

function applyStoreAccessoryItem(item) {
  if (!item?.asset_category || !item?.asset_id) {
    showToast('This store accessory is missing asset metadata.', 'warning');
    return;
  }

  const modeButton = Array.from(dom.modeBtns || []).find(btn => btn.dataset.mode === 'accessory');
  if (modeButton && state.mode !== 'accessory') {
    modeButton.click();
  } else {
    state.mode = 'accessory';
  }

  state.selectedAccessoryCategory = item.asset_category;
  state.selectedAssetId = item.asset_id;

  const categorySelect = $('editor-accessory-category');
  if (categorySelect) categorySelect.value = item.asset_category;

  const scaleInput = $('editor-accessory-scale');
  const offsetXInput = $('editor-accessory-offset-x');
  const offsetYInput = $('editor-accessory-offset-y');
  if (scaleInput) scaleInput.value = String(Number(item.fit_profile?.scale_hint ?? 1.0));
  if (offsetXInput) offsetXInput.value = String(Number(item.fit_profile?.offset_x ?? 0.0));
  if (offsetYInput) offsetYInput.value = String(Number(item.fit_profile?.offset_y ?? 0.0));

  setAccessoryRenderModeForCategory();
  const renderMode = $('editor-accessory-render-mode');
  if (renderMode && item.render_mode) renderMode.value = item.render_mode;

  upsertCurrentAccessory();
  const latest = state.selectedAccessories.find(accessory => accessory.category === item.asset_category);
  if (latest) latest.alpha = Number(item.fit_profile?.alpha ?? latest.alpha ?? 0.96);

  renderAssetPicker();
  renderAccessory3dControls();
  renderSelectedAccessories();
  updateEditorDebug();
  scheduleAutoApply('store accessory selected');
}

function renderStorePicker() {
  renderStoreTabs();
  const items = storeItemsForSelectedSlot();
  setSelectedStoreLabel();
  if (dom.tryonStoreCount) {
    const enabled = items.filter(item => item.enabled).length;
    dom.tryonStoreCount.textContent = `${enabled}/${items.length} available`;
  }
  if (!dom.tryonStoreGrid) return;
  dom.tryonStoreGrid.innerHTML = '';

  if (!items.length) {
    const empty = document.createElement('p');
    empty.className = 'col-span-full rounded border border-surface-600 px-3 py-6 text-center text-xs text-slate-500';
    empty.textContent = 'No store products in this slot yet.';
    dom.tryonStoreGrid.appendChild(empty);
    return;
  }

  items.forEach(item => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'group flex min-h-[190px] flex-col overflow-hidden rounded-md border border-surface-600 bg-surface-800 text-left transition-colors hover:border-brand-400 hover:bg-surface-700/70';
    if (item.id === state.tryon.selectedStoreItemId) {
      button.classList.add('border-brand-400', 'bg-brand-500/10');
    }
    if (!item.enabled) {
      button.classList.add('opacity-45');
    }

    const imageWrap = document.createElement('span');
    imageWrap.className = 'flex aspect-square w-full items-center justify-center bg-surface-900/60 p-2';
    if (item.thumbnail) {
      const image = document.createElement('img');
      image.src = `/${item.thumbnail}`;
      image.alt = '';
      image.loading = 'lazy';
      image.className = 'h-full w-full object-contain';
      imageWrap.appendChild(image);
    } else {
      const empty = document.createElement('span');
      empty.className = 'text-[10px] text-slate-600';
      empty.textContent = 'No image';
      imageWrap.appendChild(empty);
    }

    const label = document.createElement('span');
    label.className = 'flex min-h-[72px] flex-col gap-1 p-2';
    const title = document.createElement('span');
    title.className = 'block max-h-8 overflow-hidden text-xs font-semibold text-slate-200';
    title.textContent = item.name || item.id;
    const detail = document.createElement('span');
    detail.className = 'text-[10px] text-slate-500';
    const sourceClass = item.source_class ? ` - ${String(item.source_class).replace('-', ' ')}` : '';
    detail.textContent = `${STORE_SLOT_LABELS[item.slot] || item.slot || 'Store'}${sourceClass}`;
    label.appendChild(title);
    label.appendChild(detail);

    const tag = document.createElement('span');
    tag.className = 'mt-auto inline-flex w-fit rounded border border-brand-400/30 bg-brand-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-brand-200';
    tag.textContent = item.pipeline === 'virtual_tryon' ? 'try on' : 'apply accessory';
    label.appendChild(tag);

    button.appendChild(imageWrap);
    button.appendChild(label);
    button.addEventListener('click', () => selectStoreItem(item));
    dom.tryonStoreGrid.appendChild(button);
  });
}

function bindTryonControls() {
  if (dom.tryonStoreOpen) {
    dom.tryonStoreOpen.addEventListener('click', openStoreModal);
  }
  if (dom.tryonStoreClose) {
    dom.tryonStoreClose.addEventListener('click', closeStoreModal);
  }
  if (dom.tryonStoreModal) {
    dom.tryonStoreModal.addEventListener('click', event => {
      if (event.target === dom.tryonStoreModal) closeStoreModal();
    });
  }
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && dom.tryonStoreModal && !dom.tryonStoreModal.classList.contains('hidden')) {
      closeStoreModal();
    }
  });
  if (dom.tryonStoreClear) {
    dom.tryonStoreClear.addEventListener('click', () => {
      state.tryon.selectedStoreItemId = null;
      setSelectedStoreLabel();
      renderStorePicker();
    });
  }

  if (dom.tryonGarmentInput) {
    dom.tryonGarmentInput.addEventListener('change', () => {
      const file = dom.tryonGarmentInput.files?.[0] || null;
      state.tryon.garmentFile = file;
      if (file) state.tryon.selectedStoreItemId = null;
      renderStorePicker();
      if (dom.tryonGarmentName) {
        dom.tryonGarmentName.textContent = file
          ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB`
          : 'No garment selected.';
      }
    });
  }

  if (dom.tryonModelType) {
    dom.tryonModelType.addEventListener('change', () => {
      state.tryon.modelType = dom.tryonModelType.value || 'dc';
      syncTryonCategoryOptions();
      if (state.tryon.modelType === 'hd' && dom.tryonCategory) {
        state.tryon.category = 'upperbody';
        dom.tryonCategory.value = 'upperbody';
      }
    });
  }
  if (dom.tryonCategory) {
    dom.tryonCategory.addEventListener('change', () => {
      state.tryon.category = dom.tryonCategory.value || 'upperbody';
    });
  }
  if (dom.tryonSteps) {
    dom.tryonSteps.addEventListener('input', () => {
      state.tryon.steps = Number(dom.tryonSteps.value) || 20;
      if (dom.tryonStepsValue) dom.tryonStepsValue.textContent = String(state.tryon.steps);
    });
  }
  if (dom.tryonScale) {
    dom.tryonScale.addEventListener('input', () => {
      state.tryon.scale = Number(dom.tryonScale.value) || 2.0;
      if (dom.tryonScaleValue) dom.tryonScaleValue.textContent = state.tryon.scale.toFixed(1);
    });
  }
}

function syncTryonCategoryOptions() {
  if (!dom.tryonCategory) return;
  const hdOnly = state.tryon.modelType === 'hd';
  Array.from(dom.tryonCategory.options).forEach(option => {
    option.disabled = hdOnly && option.value !== 'upperbody';
  });
}

function updateParamSnapshot() {
  dom.snapMode.textContent      = state.mode;
  const isAiAging = state.mode === 'aging' && state.agingAlgorithm === 'ai';
  dom.snapAging.textContent     = isAiAging ? fmt(state.agingIntensity, 0) : fmt(state.agingIntensity, 2);
  dom.snapSmile.textContent     = fmt(state.smileIntensity);
  dom.snapEyebrow.textContent   = fmt(state.eyebrowHeight);
  dom.snapSlim.textContent      = fmt(state.faceSlimming);
  dom.snapLandmarks.textContent = state.showLandmarks ? 'on' : 'off';
  dom.snapGrayscale.textContent = state.showGrayscale ? 'on' : 'off';
}

/* ═══════════════════════════════════════════════
   Mode selector
═══════════════════════════════════════════════ */

dom.modeBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    let hadEffects = false;
    if (state.mode === 'accessory' && btn.dataset.mode !== 'accessory') {
      hadEffects = state.selectedAccessories.length > 0 ||
        editorEnabled('editor-hair-enabled') ||
        editorEnabled('editor-eye-enabled') ||
        editorEnabled('editor-skin-smooth-enabled') ||
        editorEnabled('editor-lipstick-enabled') ||
        editorEnabled('editor-blush-enabled') ||
        editorEnabled('editor-eyeshadow-enabled') ||
        editorEnabled('editor-eyeliner-enabled');
      clearAccessoryAndMakeupState();
    }

    state.mode = btn.dataset.mode;
    // Show/hide parameter groups based on active mode
    dom.groupAging.style.display       = state.mode === 'aging'       ? '' : 'none';
    dom.groupExpression.style.display  = state.mode === 'expression'  ? '' : 'none';
    dom.groupAccessory.style.display   = state.mode === 'accessory'   ? '' : 'none';
    if (dom.groupTryon) {
      dom.groupTryon.style.display = state.mode === 'virtual_tryon' ? '' : 'none';
    }
    if (dom.aiExpressionPanel) {
      dom.aiExpressionPanel.style.display = state.mode === 'ai_expression' ? '' : 'none';
    }

    const modeLabels = {
      expression: 'Expression',
      aging: 'Aging',
      accessory: 'Accessory',
      ai_expression: 'AI Expression',
      virtual_tryon: 'Virtual Try-On',
    };
    dom.modeBadge.textContent = modeLabels[state.mode] ?? state.mode;

    // Style active mode button with correct accent colour
    dom.modeBtns.forEach(b => {
      const isActive = b.dataset.mode === state.mode;
      const isAccessory = b.dataset.mode === 'accessory';
      const isAiExpression = b.dataset.mode === 'ai_expression';
      b.classList.toggle('border-brand-500',   isActive && (!isAccessory || isAiExpression));
      b.classList.toggle('bg-brand-600/20',    isActive && (!isAccessory || isAiExpression));
      b.classList.toggle('text-brand-300',     isActive && (!isAccessory || isAiExpression));
      b.classList.toggle('border-purple-500',  isActive && isAccessory);
      b.classList.toggle('bg-purple-600/20',   isActive && isAccessory);
      b.classList.toggle('text-purple-300',    isActive && isAccessory);
      b.classList.toggle('border-surface-500', !isActive);
      b.classList.toggle('bg-transparent',     !isActive);
      b.classList.toggle('text-slate-400',     !isActive);
      b.setAttribute('aria-pressed', String(isActive));
    });

    updateParamSnapshot();

    if (hadEffects && state.mode !== 'virtual_tryon') {
      processImage();
    }
  });
});

/* ═══════════════════════════════════════════════
   Accessory item selector
═══════════════════════════════════════════════ */
dom.accBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    state.accessoryItem = btn.dataset.item;
    state.editorTab = state.accessoryItem === 'makeup' ? 'makeup' : 'accessories';
    dom.accBtns.forEach(b => {
      const active = b.dataset.item === state.accessoryItem;
      b.classList.toggle('border-purple-500',  active);
      b.classList.toggle('bg-purple-600/20',   active);
      b.classList.toggle('text-purple-300',    active);
      b.classList.toggle('border-surface-500', !active);
      b.classList.toggle('bg-transparent',     !active);
      b.classList.toggle('text-slate-400',     !active);
    });
    renderEditorTab();
    updateEditorDebug();
  });
});

/* ═══════════════════════════════════════════════
   Aging Algorithm Selector
═══════════════════════════════════════════════ */
if (dom.algoFreq && dom.algoAi) {
    dom.algoFreq.addEventListener('click', () => {
        state.agingAlgorithm = 'frequency';
        dom.algoFreq.className = 'flex-1 py-1.5 text-[10px] font-semibold rounded-md bg-brand-500/20 text-brand-300 border border-brand-500/30 transition-all shadow-sm';
        dom.algoAi.className = 'flex-1 py-1.5 text-[10px] font-semibold rounded-md text-slate-400 border border-transparent hover:text-slate-200 transition-all';
        state.agingIntensity = 1.0;
        if (dom.sliderAging) {
            dom.sliderAging.min = '0.0';
            dom.sliderAging.max = '2.0';
            dom.sliderAging.step = '0.05';
            dom.sliderAging.value = '1.0';
        }
        if (dom.lblSliderAging) {
            dom.lblSliderAging.textContent = 'Aging Intensity';
        }
        if (dom.valAging) {
            dom.valAging.textContent = '1.00';
        }
        if (dom.labelsSliderAging) {
            dom.labelsSliderAging.innerHTML = `
                <span>0.0 (De-Aging)</span>
                <span>1.0 (Base)</span>
                <span>2.0 (Aging)</span>
            `;
        }
        updateParamSnapshot();
    });
    dom.algoAi.addEventListener('click', () => {
        state.agingAlgorithm = 'ai';
        dom.algoAi.className = 'flex-1 py-1.5 text-[10px] font-semibold rounded-md bg-purple-500/20 text-purple-300 border border-purple-500/30 transition-all shadow-sm';
        dom.algoFreq.className = 'flex-1 py-1.5 text-[10px] font-semibold rounded-md text-slate-400 border border-transparent hover:text-slate-200 transition-all';
        state.agingIntensity = 60.0;
        if (dom.sliderAging) {
            dom.sliderAging.min = '7';
            dom.sliderAging.max = '85';
            dom.sliderAging.step = '1';
            dom.sliderAging.value = '60';
        }
        if (dom.lblSliderAging) {
            dom.lblSliderAging.textContent = 'Target Age';
        }
        if (dom.valAging) {
            dom.valAging.textContent = '60';
        }
        if (dom.labelsSliderAging) {
            dom.labelsSliderAging.innerHTML = `
                <span>7 (Childhood)</span>
                <span>35 (Adult)</span>
                <span>85 (Senior)</span>
            `;
        }
        updateParamSnapshot();
    });
}


/* ═══════════════════════════════════════════════
   Landmarks toggle
═══════════════════════════════════════════════ */
dom.landmarkToggle.addEventListener('change', () => {
  state.showLandmarks = dom.landmarkToggle.checked;
  updateParamSnapshot();
  if (state.file && dom.imgOriginal.src) {
    toggleLandmarkCanvas();
  }
});

function toggleLandmarkCanvas() {
  if (!state.resultData?.landmark_detection?.points) return;
  dom.canvasLmOrig.style.zIndex = '10';
  dom.canvasLmOrig.classList.toggle('hidden', !state.showLandmarks);
}

/* ═══════════════════════════════════════════════
   Grayscale toggle
═══════════════════════════════════════════════ */
dom.grayscaleToggle.addEventListener('change', () => {
  state.showGrayscale = dom.grayscaleToggle.checked;
  updateParamSnapshot();
  applyGrayscaleOverlay();
});

/**
 * Show or hide the grayscale overlay images on both stages.
 */
function applyGrayscaleOverlay() {
  const hasData = Boolean(state.grayscaleOrigUrl);

  // Original stage overlay
  if (hasData && state.showGrayscale) {
    dom.imgGrayscaleOrig.classList.remove('hidden');
    dom.imgGrayscaleOrig.classList.add('visible');
  } else {
    dom.imgGrayscaleOrig.classList.remove('visible');
    // Keep hidden class so it doesn't occupy layout space when no data
    if (!hasData) dom.imgGrayscaleOrig.classList.add('hidden');
    else          dom.imgGrayscaleOrig.classList.add('hidden');
  }

  // Result stage overlay (only if result grayscale is available)
  const hasResultData = Boolean(state.grayscaleResUrl);
  if (hasResultData && state.showGrayscale) {
    dom.imgGrayscaleResult.classList.remove('hidden');
    dom.imgGrayscaleResult.classList.add('visible');
  } else {
    dom.imgGrayscaleResult.classList.remove('visible');
    dom.imgGrayscaleResult.classList.add('hidden');
  }

  // Badge in header
  if (state.showGrayscale && hasData) {
    dom.grayscaleBadge.classList.remove('hidden');
    dom.grayscaleBadge.classList.add('flex');
  } else {
    dom.grayscaleBadge.classList.add('hidden');
    dom.grayscaleBadge.classList.remove('flex');
  }
}

function setGrayscaleData(origPath, resultPath) {
  if (origPath) {
    state.grayscaleOrigUrl = '/' + origPath; 
    dom.imgGrayscaleOrig.src = state.grayscaleOrigUrl;
    dom.grayscaleNote.classList.remove('hidden');
  }
  if (resultPath) {
    state.grayscaleResUrl = '/' + resultPath;
    dom.imgGrayscaleResult.src = state.grayscaleResUrl;
  }
  applyGrayscaleOverlay();
}

function clearGrayscaleData() {
  state.grayscaleOrigUrl   = null;
  state.grayscaleResUrl    = null;
  dom.imgGrayscaleOrig.src  = '';
  dom.imgGrayscaleResult.src = '';
  dom.imgGrayscaleOrig.classList.add('hidden');
  dom.imgGrayscaleResult.classList.add('hidden');
  dom.imgGrayscaleOrig.classList.remove('visible');
  dom.imgGrayscaleResult.classList.remove('visible');
  dom.grayscaleBadge.classList.add('hidden');
  dom.grayscaleBadge.classList.remove('flex');
  dom.grayscaleNote.classList.add('hidden');
}

/* ═══════════════════════════════════════════════
   Dropzone & file handling
═══════════════════════════════════════════════ */
['dragenter', 'dragover'].forEach(evt => {
  dom.dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dom.dropzone.classList.add('drag-over');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dom.dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dom.dropzone.classList.remove('drag-over');
  });
});
dom.dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});
dom.fileInput.addEventListener('change', () => {
  const f = dom.fileInput.files?.[0];
  if (f) handleFile(f);
  dom.fileInput.value = '';
});

function isAllowedImageFile(file) {
  const allowed = new Set(['image/jpeg', 'image/jpg', 'image/png']);
  if (file.type && allowed.has(file.type)) return true;
  if (file.type && file.type !== '') return false;
  return /\.(jpe?g|png)$/i.test(file.name || '');
}

/**
 * Show a spectrum image in an <img> element.
 * src may be a data URL or a server path.
 * onErrorFallback is called if the image fails to load.
 */
function showFFTImage(targetImg, placeholder, src, onErrorFallback) {
  if (!src) {
    targetImg.src = '';
    targetImg.classList.add('hidden');
    placeholder.classList.remove('hidden');
    return;
  }
  targetImg.classList.add('hidden');
  placeholder.classList.remove('hidden');
  targetImg.onload = () => {
    targetImg.classList.remove('hidden');
    placeholder.classList.add('hidden');
  };
  targetImg.onerror = () => {
    if (typeof onErrorFallback === 'function') { onErrorFallback(); return; }
    targetImg.classList.add('hidden');
    placeholder.textContent = 'Spectrum unavailable';
    placeholder.classList.remove('hidden');
  };
  targetImg.src = src;
}

/** Reset both FFT panels to "No data" state. */
function clearFFTData() {
  [dom.imgFftOrig, dom.imgFftResult, dom.imgFftPhaseOrig, dom.imgFftPhaseResult].forEach(img => {
    if (img) {
      img.src = '';
      img.classList.add('hidden');
    }
  });
  [dom.fftOrigPholder, dom.fftResultPholder, dom.fftPhaseOrigPholder, dom.fftPhaseResultPholder].forEach(ph => {
    if (ph) {
      ph.textContent = 'No data';
      ph.classList.remove('hidden');
    }
  });
}

const PLASMA_MAP = [[12, 7, 134], [16, 7, 135], [19, 6, 137], [21, 6, 138], [24, 6, 139], [27, 6, 140], [29, 6, 141], [31, 5, 142], [33, 5, 143], [35, 5, 144], [37, 5, 145], [39, 5, 146], [41, 5, 147], [43, 5, 148], [45, 4, 148], [47, 4, 149], [49, 4, 150], [51, 4, 151], [52, 4, 152], [54, 4, 152], [56, 4, 153], [58, 4, 154], [59, 3, 154], [61, 3, 155], [63, 3, 156], [64, 3, 156], [66, 3, 157], [68, 3, 158], [69, 3, 158], [71, 2, 159], [73, 2, 159], [74, 2, 160], [76, 2, 161], [78, 2, 161], [79, 2, 162], [81, 1, 162], [82, 1, 163], [84, 1, 163], [86, 1, 163], [87, 1, 164], [89, 1, 164], [90, 0, 165], [92, 0, 165], [94, 0, 165], [95, 0, 166], [97, 0, 166], [98, 0, 166], [100, 0, 167], [101, 0, 167], [103, 0, 167], [104, 0, 167], [106, 0, 167], [108, 0, 168], [109, 0, 168], [111, 0, 168], [112, 0, 168], [114, 0, 168], [115, 0, 168], [117, 0, 168], [118, 1, 168], [120, 1, 168], [121, 1, 168], [123, 2, 168], [124, 2, 167], [126, 3, 167], [127, 3, 167], [129, 4, 167], [130, 4, 167], [132, 5, 166], [133, 6, 166], [134, 7, 166], [136, 7, 165], [137, 8, 165], [139, 9, 164], [140, 10, 164], [142, 12, 164], [143, 13, 163], [144, 14, 163], [146, 15, 162], [147, 16, 161], [149, 17, 161], [150, 18, 160], [151, 19, 160], [153, 20, 159], [154, 21, 158], [155, 23, 158], [157, 24, 157], [158, 25, 156], [159, 26, 155], [160, 27, 155], [162, 28, 154], [163, 29, 153], [164, 30, 152], [165, 31, 151], [167, 33, 151], [168, 34, 150], [169, 35, 149], [170, 36, 148], [172, 37, 147], [173, 38, 146], [174, 39, 145], [175, 40, 144], [176, 42, 143], [177, 43, 143], [178, 44, 142], [180, 45, 141], [181, 46, 140], [182, 47, 139], [183, 48, 138], [184, 50, 137], [185, 51, 136], [186, 52, 135], [187, 53, 134], [188, 54, 133], [189, 55, 132], [190, 56, 131], [191, 57, 130], [192, 59, 129], [193, 60, 128], [194, 61, 128], [195, 62, 127], [196, 63, 126], [197, 64, 125], [198, 65, 124], [199, 66, 123], [200, 68, 122], [201, 69, 121], [202, 70, 120], [203, 71, 119], [204, 72, 118], [205, 73, 117], [206, 74, 117], [207, 75, 116], [208, 77, 115], [209, 78, 114], [209, 79, 113], [210, 80, 112], [211, 81, 111], [212, 82, 110], [213, 83, 109], [214, 85, 109], [215, 86, 108], [215, 87, 107], [216, 88, 106], [217, 89, 105], [218, 90, 104], [219, 91, 103], [220, 93, 102], [220, 94, 102], [221, 95, 101], [222, 96, 100], [223, 97, 99], [223, 98, 98], [224, 100, 97], [225, 101, 96], [226, 102, 96], [227, 103, 95], [227, 104, 94], [228, 106, 93], [229, 107, 92], [229, 108, 91], [230, 109, 90], [231, 110, 90], [232, 112, 89], [232, 113, 88], [233, 114, 87], [234, 115, 86], [234, 116, 85], [235, 118, 84], [236, 119, 84], [236, 120, 83], [237, 121, 82], [237, 123, 81], [238, 124, 80], [239, 125, 79], [239, 126, 78], [240, 128, 77], [240, 129, 77], [241, 130, 76], [242, 132, 75], [242, 133, 74], [243, 134, 73], [243, 135, 72], [244, 137, 71], [244, 138, 71], [245, 139, 70], [245, 141, 69], [246, 142, 68], [246, 143, 67], [246, 145, 66], [247, 146, 65], [247, 147, 65], [248, 149, 64], [248, 150, 63], [248, 152, 62], [249, 153, 61], [249, 154, 60], [250, 156, 59], [250, 157, 58], [250, 159, 58], [250, 160, 57], [251, 162, 56], [251, 163, 55], [251, 164, 54], [252, 166, 53], [252, 167, 53], [252, 169, 52], [252, 170, 51], [252, 172, 50], [252, 173, 49], [253, 175, 49], [253, 176, 48], [253, 178, 47], [253, 179, 46], [253, 181, 45], [253, 182, 45], [253, 184, 44], [253, 185, 43], [253, 187, 43], [253, 188, 42], [253, 190, 41], [253, 192, 41], [253, 193, 40], [253, 195, 40], [253, 196, 39], [253, 198, 38], [252, 199, 38], [252, 201, 38], [252, 203, 37], [252, 204, 37], [252, 206, 37], [251, 208, 36], [251, 209, 36], [251, 211, 36], [250, 213, 36], [250, 214, 36], [250, 216, 36], [249, 217, 36], [249, 219, 36], [248, 221, 36], [248, 223, 36], [247, 224, 36], [247, 226, 37], [246, 228, 37], [246, 229, 37], [245, 231, 38], [245, 233, 38], [244, 234, 38], [243, 236, 38], [243, 238, 38], [242, 240, 38], [242, 241, 38], [241, 243, 38], [240, 245, 37], [240, 246, 35], [239, 248, 33]];
const TWILIGHT_MAP = [[225, 216, 226], [224, 217, 226], [223, 217, 225], [222, 217, 224], [221, 217, 224], [219, 216, 223], [217, 216, 222], [216, 215, 221], [214, 214, 220], [212, 214, 219], [210, 213, 218], [207, 212, 217], [205, 210, 216], [202, 209, 215], [199, 208, 214], [197, 207, 212], [194, 205, 211], [191, 204, 210], [188, 202, 209], [185, 201, 208], [182, 199, 207], [179, 198, 206], [176, 196, 205], [173, 195, 204], [170, 193, 203], [167, 192, 202], [164, 190, 202], [161, 188, 201], [158, 187, 200], [155, 185, 200], [152, 183, 199], [150, 181, 198], [147, 180, 198], [146, 179, 198], [142, 176, 197], [139, 174, 197], [137, 172, 196], [136, 171, 196], [132, 169, 195], [130, 167, 195], [128, 165, 195], [127, 164, 194], [124, 161, 194], [122, 159, 194], [120, 157, 193], [119, 156, 193], [116, 154, 193], [115, 152, 192], [113, 150, 192], [112, 149, 192], [110, 146, 191], [109, 144, 191], [107, 142, 191], [107, 141, 191], [105, 137, 190], [104, 135, 190], [103, 133, 189], [102, 132, 189], [101, 129, 189], [100, 127, 188], [100, 125, 188], [99, 124, 187], [98, 120, 187], [98, 118, 186], [97, 116, 186], [97, 114, 185], [97, 113, 185], [96, 109, 184], [96, 107, 183], [95, 105, 182], [95, 103, 182], [95, 100, 181], [95, 98, 180], [95, 96, 179], [95, 94, 179], [94, 81, 177], [94, 89, 176], [94, 86, 175], [94, 84, 174], [94, 81, 173], [94, 79, 172], [94, 77, 170], [94, 75, 170], [93, 72, 167], [93, 69, 166], [93, 67, 164], [93, 64, 163], [93, 62, 161], [92, 60, 159], [92, 57, 157], [92, 56, 156], [91, 52, 153], [91, 50, 151], [90, 48, 149], [90, 45, 146], [89, 43, 144], [89, 41, 141], [88, 39, 139], [87, 37, 137], [86, 34, 133], [85, 33, 130], [84, 31, 127], [83, 29, 124], [82, 27, 120], [80, 26, 117], [79, 25, 114], [78, 24, 112], [76, 22, 107], [74, 21, 103], [73, 21, 100], [71, 20, 96], [69, 19, 93], [68, 18, 90], [66, 18, 87], [65, 18, 85], [62, 17, 81], [61, 17, 78], [59, 17, 75], [58, 16, 72], [56, 16, 70], [55, 16, 67], [54, 16, 65], [53, 16, 64], [51, 17, 61], [50, 17, 59], [50, 17, 58], [48, 18, 56], [47, 19, 55], [47, 19, 54], [49, 18, 54], [50, 18, 55], [51, 17, 55], [52, 17, 55], [54, 17, 56], [55, 17, 57], [57, 17, 57], [59, 17, 58], [61, 17, 59], [63, 17, 60], [65, 17, 61], [67, 18, 62], [69, 18, 63], [71, 18, 64], [74, 19, 65], [76, 19, 66], [79, 20, 67], [81, 20, 68], [84, 21, 69], [85, 21, 70], [89, 22, 71], [92, 22, 72], [94, 23, 73], [97, 24, 74], [99, 24, 75], [102, 25, 76], [105, 26, 76], [107, 26, 77], [110, 27, 78], [113, 28, 78], [115, 29, 78], [118, 30, 79], [120, 31, 79], [123, 32, 79], [125, 33, 80], [127, 34, 80], [130, 36, 80], [132, 37, 80], [135, 39, 80], [137, 40, 80], [139, 42, 80], [141, 44, 80], [144, 45, 80], [146, 47, 79], [148, 49, 79], [150, 50, 79], [152, 52, 79], [154, 54, 79], [155, 56, 79], [157, 58, 79], [159, 60, 79], [160, 61, 79], [162, 64, 79], [164, 66, 79], [166, 68, 79], [167, 70, 79], [169, 73, 80], [170, 75, 80], [172, 77, 80], [173, 79, 80], [175, 81, 81], [176, 84, 81], [178, 86, 82], [179, 88, 82], [180, 90, 83], [181, 93, 83], [182, 95, 84], [183, 96, 84], [185, 100, 86], [186, 102, 87], [187, 104, 87], [188, 107, 89], [189, 109, 90], [190, 112, 91], [191, 114, 92], [192, 116, 93], [192, 119, 95], [193, 121, 96], [194, 124, 98], [195, 126, 100], [195, 129, 102], [196, 131, 104], [197, 134, 106], [197, 135, 107], [198, 139, 110], [199, 141, 112], [199, 143, 114], [200, 146, 117], [200, 148, 120], [201, 151, 122], [201, 153, 125], [202, 156, 128], [202, 158, 131], [203, 161, 133], [204, 163, 137], [204, 165, 140], [205, 168, 143], [205, 170, 146], [206, 172, 149], [206, 174, 151], [207, 177, 156], [208, 179, 159], [209, 182, 163], [210, 184, 166], [211, 186, 169], [211, 188, 173], [212, 190, 176], [213, 192, 180], [214, 194, 183], [215, 196, 187], [216, 198, 190], [217, 200, 193], [218, 202, 196], [219, 204, 200], [219, 206, 203], [220, 206, 204], [221, 209, 208], [222, 210, 211], [222, 211, 213], [223, 213, 215], [223, 214, 217], [224, 214, 219], [224, 215, 221], [225, 216, 222], [225, 216, 223], [225, 216, 225], [225, 216, 225]];

function getPlasmaColor(v) {
  const idx = Math.min(255, Math.max(0, Math.floor(v * 256)));
  return PLASMA_MAP[idx];
}

function getTwilightColor(v) {
  const idx = Math.min(255, Math.max(0, Math.floor(v * 256)));
  return TWILIGHT_MAP[idx];
}

/**
 * Compute an in-browser FFT magnitude spectrum from an <img> element
 * and return it as a PNG data URL.
 * Throws if the element has no loaded image.
 */
function computeFFTDataUrl(imgEl) {
  if (!imgEl || !imgEl.naturalWidth || !imgEl.naturalHeight) {
    throw new Error('Image not loaded or has zero dimensions.');
  }
  const N   = 256;
  const off = document.createElement('canvas');
  off.width = off.height = N;
  off.getContext('2d').drawImage(imgEl, 0, 0, N, N);
  const pixels = off.getContext('2d').getImageData(0, 0, N, N).data;

  const gray = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) {
    gray[i] = (pixels[i*4]*0.299 + pixels[i*4+1]*0.587 + pixels[i*4+2]*0.114) / 255;
  }

  const mag = fft2DMag(gray, N);
  const out = document.createElement('canvas');
  out.width = out.height = N;
  drawFFTMagnitude(mag, N, out);
  return out.toDataURL('image/png');
}

/**
 * Compute 2-D DFT magnitude with fftshift via a naive O(N^3) implementation.
 * Returns a Float32Array of log(1 + magnitude) values, shifted so DC is centred.
 */
function fft2DMag(gray, N) {
  const mag   = new Float32Array(N * N);
  const TAU   = 2 * Math.PI;
  const rowRe = new Float32Array(N * N);
  const rowIm = new Float32Array(N * N);

  // Row-wise DFT
  for (let r = 0; r < N; r++) {
    for (let k = 0; k < N; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < N; n++) {
        const angle = TAU * k * n / N;
        re += gray[r * N + n] * Math.cos(angle);
        im -= gray[r * N + n] * Math.sin(angle);
      }
      rowRe[r * N + k] = re;
      rowIm[r * N + k] = im;
    }
  }

  // Column-wise DFT + fftshift + log(1 + magnitude)
  for (let col = 0; col < N; col++) {
    for (let k = 0; k < N; k++) {
      let re = 0, im = 0;
      for (let r = 0; r < N; r++) {
        const angle = TAU * k * r / N;
        re += rowRe[r * N + col] * Math.cos(angle) + rowIm[r * N + col] * Math.sin(angle);
        im += rowIm[r * N + col] * Math.cos(angle) - rowRe[r * N + col] * Math.sin(angle);
      }
      // fftshift: move DC to centre
      const sr = (k + N / 2) % N;
      const sc = (col + N / 2) % N;
      mag[sr * N + sc] = Math.log(1 + Math.sqrt(re * re + im * im));
    }
  }
  return mag;
}

/**
 * Render a magnitude spectrum (Float32Array, shifted) onto a canvas
 * using Matplotlib's plasma colormap.
 */
function drawFFTMagnitude(mag, N, targetCanvas) {
  const ctx   = targetCanvas.getContext('2d');
  const W     = targetCanvas.width;
  const H     = targetCanvas.height;
  const imgData = ctx.createImageData(W, H);
  const scale = N / W;

  let minVal = mag[0];
  let maxVal = mag[0];
  for (let i = 1; i < mag.length; i++) {
    if (mag[i] < minVal) minVal = mag[i];
    if (mag[i] > maxVal) maxVal = mag[i];
  }
  const range = maxVal - minVal;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const val = mag[Math.floor(y * scale) * N + Math.floor(x * scale)];
      const v = range > 0 ? (val - minVal) / range : 0;
      const rgb = getPlasmaColor(v);
      const idx = (y * W + x) * 4;
      imgData.data[idx]   = rgb[0];
      imgData.data[idx+1] = rgb[1];
      imgData.data[idx+2] = rgb[2];
      imgData.data[idx+3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

/**
 * Compute 2-D DFT phase with fftshift via naive implementation.
 * Returns a Float32Array of phase values in [-pi, pi], shifted so DC is centred.
 */
function fft2DPhase(gray, N) {
  const phase = new Float32Array(N * N);
  const TAU   = 2 * Math.PI;
  const rowRe = new Float32Array(N * N);
  const rowIm = new Float32Array(N * N);

  // Row-wise DFT
  for (let r = 0; r < N; r++) {
    for (let k = 0; k < N; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < N; n++) {
        const angle = TAU * k * n / N;
        re += gray[r * N + n] * Math.cos(angle);
        im -= gray[r * N + n] * Math.sin(angle);
      }
      rowRe[r * N + k] = re;
      rowIm[r * N + k] = im;
    }
  }

  // Column-wise DFT + fftshift + Phase angle
  for (let col = 0; col < N; col++) {
    for (let k = 0; k < N; k++) {
      let re = 0, im = 0;
      for (let r = 0; r < N; r++) {
        const angle = TAU * k * r / N;
        re += rowRe[r * N + col] * Math.cos(angle) + rowIm[r * N + col] * Math.sin(angle);
        im += rowIm[r * N + col] * Math.cos(angle) - rowRe[r * N + col] * Math.sin(angle);
      }
      // fftshift: move DC to centre
      const sr = (k + N / 2) % N;
      const sc = (col + N / 2) % N;
      phase[sr * N + sc] = Math.atan2(im, re);
    }
  }
  return phase;
}

/** Helper for HSL to RGB conversion */
function hslToRgb(h, s, l) {
  let r, g, b;
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };

  if (s === 0) {
    r = g = b = l; // achromatic
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Render a phase spectrum onto a canvas using Twilight colormap.
 */
function drawFFTPhase(phase, N, targetCanvas) {
  const ctx   = targetCanvas.getContext('2d');
  const W     = targetCanvas.width;
  const H     = targetCanvas.height;
  const imgData = ctx.createImageData(W, H);
  const scale = N / W;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = phase[Math.floor(y * scale) * N + Math.floor(x * scale)];
      const v = (p + Math.PI) / (2 * Math.PI); // map [-pi, pi] to [0, 1]
      const rgb = getTwilightColor(v);
      const idx = (y * W + x) * 4;
      imgData.data[idx]   = rgb[0];
      imgData.data[idx+1] = rgb[1];
      imgData.data[idx+2] = rgb[2];
      imgData.data[idx+3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

/**
 * Compute an in-browser FFT phase diagram from an <img> element
 * and return it as a PNG data URL.
 */
function computeFFTPhaseDataUrl(imgEl) {
  if (!imgEl || !imgEl.naturalWidth || !imgEl.naturalHeight) {
    throw new Error('Image not loaded or has zero dimensions.');
  }
  const N   = 256;
  const off = document.createElement('canvas');
  off.width = off.height = N;
  off.getContext('2d').drawImage(imgEl, 0, 0, N, N);
  const pixels = off.getContext('2d').getImageData(0, 0, N, N).data;

  const gray = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) {
    gray[i] = (pixels[i*4]*0.299 + pixels[i*4+1]*0.587 + pixels[i*4+2]*0.114) / 255;
  }

  const phase = fft2DPhase(gray, N);
  const out = document.createElement('canvas');
  out.width = out.height = N;
  drawFFTPhase(phase, N, out);
  return out.toDataURL('image/png');
}

/**
 * When imgEl finishes loading, compute FFT from it and show the
 * result in targetImg / placeholder.
 */
function runFftWhenImageReady(imgEl, targetImg, placeholder, isPhase = false) {
  const run = () => {
    try {
      const dataUrl = isPhase ? computeFFTPhaseDataUrl(imgEl) : computeFFTDataUrl(imgEl);
      showFFTImage(targetImg, placeholder, dataUrl);
    } catch (err) {
      console.error('[FaceWarp] FFT:', err);
      targetImg.classList.add('hidden');
      placeholder.textContent = isPhase ? 'Phase diagram unavailable' : 'Spectrum unavailable';
      placeholder.classList.remove('hidden');
    }
  };
  if (imgEl.decode) {
    imgEl.decode().then(run).catch(run);
  } else if (imgEl.complete && imgEl.naturalWidth) {
    run();
  } else {
    imgEl.addEventListener('load', run, { once: true });
  }
}

function handleFile(file) {
  if (!isAllowedImageFile(file)) {
    showToast('Only JPG and PNG files are supported.', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File exceeds 10 MB limit.', 'error');
    return;
  }

  state.file = file;
  state.sessionId = null;
  state.imageId = null;
  dom.btnProcess.disabled = true;
  if (dom.btnAiExpression) dom.btnAiExpression.disabled = true;
  renderAiExpressionDebug(null);

  clearGrayscaleData();
  clearAccessoryAndMakeupState();

  state.smileIntensity  = 0.0;
  state.eyebrowHeight   = 0.0;
  state.lipIntensity    = 0.0;
  state.faceSlimming    = 0.0;
  state.agingIntensity  = 1.0;
  state.agingAlgorithm  = 'frequency';

  if (dom.algoFreq && dom.algoAi) {
      dom.algoFreq.className = 'flex-1 py-1.5 text-[10px] font-semibold rounded-md bg-brand-500/20 text-brand-300 border border-brand-500/30 transition-all shadow-sm';
      dom.algoAi.className = 'flex-1 py-1.5 text-[10px] font-semibold rounded-md text-slate-400 border border-transparent hover:text-slate-200 transition-all';
  }

  if (dom.sliderAging) {
      dom.sliderAging.min = '0.0';
      dom.sliderAging.max = '2.0';
      dom.sliderAging.step = '0.05';
      dom.sliderAging.value = '1.0';
  }
  if (dom.lblSliderAging) {
      dom.lblSliderAging.textContent = 'Aging Intensity';
  }
  if (dom.labelsSliderAging) {
      dom.labelsSliderAging.innerHTML = `
          <span>0.0 (De-Aging)</span>
          <span>1.0 (Base)</span>
          <span>2.0 (Aging)</span>
      `;
  }

  dom.sliderSmile.value   = 0;
  dom.sliderEyebrow.value = 0;
  if (dom.sliderLip) dom.sliderLip.value = 0;
  dom.sliderSlim.value    = 0;
  dom.sliderAging.value   = 1;

  dom.valSmile.textContent   = '0.00';
  dom.valEyebrow.textContent = '0.00';
  if (dom.valLip) dom.valLip.textContent = '0.00';
  dom.valSlim.textContent    = '0.00';
  dom.valAging.textContent   = '1.00';

  updateParamSnapshot();

  // Matplotlib resimlerini ve placeholder'ları sıfırla
  if (dom.imgFftOrig) {
      dom.imgFftOrig.src = '';
      dom.imgFftOrig.classList.add('hidden');
  }
  if (dom.imgFftResult) {
      dom.imgFftResult.src = '';
      dom.imgFftResult.classList.add('hidden');
  }
  if (dom.imgFftPhaseOrig) {
      dom.imgFftPhaseOrig.src = '';
      dom.imgFftPhaseOrig.classList.add('hidden');
  }
  if (dom.imgFftPhaseResult) {
      dom.imgFftPhaseResult.src = '';
      dom.imgFftPhaseResult.classList.add('hidden');
  }
  if (dom.fftOrigPholder) dom.fftOrigPholder.classList.remove('hidden');
  if (dom.fftResultPholder) dom.fftResultPholder.classList.remove('hidden');
  if (dom.fftPhaseOrigPholder) dom.fftPhaseOrigPholder.classList.remove('hidden');
  if (dom.fftPhaseResultPholder) dom.fftPhaseResultPholder.classList.remove('hidden');

  const reader = new FileReader();
  reader.onerror = () => { showToast('Could not read the file from disk.', 'error'); };
  reader.onload = e => {
    const dataUrl = e.target.result;
    dom.previewThumb.src        = dataUrl;
    dom.previewName.textContent = file.name;
    dom.dropzonePreview.classList.remove('hidden');
    dom.dropzoneIcon.classList.add('hidden');

    const img   = dom.imgOriginal;
    const stage = img.closest('.img-stage');
    stage?.querySelector('.img-placeholder')?.classList.add('hidden');
    img.classList.remove('hidden');
    dom.labelOriginal.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;

    img.removeAttribute('src');
    img.src = dataUrl;
    clearFFTData();
    runFftWhenImageReady(img, dom.imgFftOrig, dom.fftOrigPholder);
    runFftWhenImageReady(img, dom.imgFftPhaseOrig, dom.fftPhaseOrigPholder, true);
  };
  reader.readAsDataURL(file);

  (async () => {
    setStatus('Uploading…', 'processing');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const j   = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = j?.error?.message ?? (typeof j?.error === 'string' ? j.error : null) ?? (Array.isArray(j?.detail) ? j.detail[0]?.msg : j?.detail) ?? `HTTP ${res.status}`;
        throw new Error(msg);
      }
      state.sessionId = j.session_id;
      state.imageId   = j.image_id;
      if (j.metadata?.width && j.metadata?.height) {
        dom.labelOriginal.textContent = `${file.name} · ${j.metadata.width}×${j.metadata.height} px · uploaded`;
      }
      dom.btnProcess.disabled = false;
      if (dom.btnAiExpression) dom.btnAiExpression.disabled = false;
      setStatus('Ready', 'idle');
      showToast('Image uploaded', 'success');
    } catch (err) {
      console.error('[FaceWarp] Upload error:', err);
      state.sessionId = null;
      state.imageId   = null;
      dom.btnProcess.disabled = true;
      if (dom.btnAiExpression) dom.btnAiExpression.disabled = true;
      setStatus('Ready (local preview only)', 'idle');
      showToast(
        (err.message || 'Server upload failed.') + ' You can still see the image; start the API to enable Process.',
        'warning',
        6000
      );
    }
  })();
}

/* ═══════════════════════════════════════════════
   Process button
═══════════════════════════════════════════════ */

function setWebcamVisible(visible) {
  if (dom.mainStageRow) dom.mainStageRow.classList.toggle('hidden', visible);
  if (dom.webcamStage) {
    dom.webcamStage.classList.toggle('hidden', !visible);
    dom.webcamStage.classList.toggle('flex', visible);
  }
  if (dom.webcamSpinner) dom.webcamSpinner.classList.toggle('hidden', !visible);
}

function drawVideoToCanvas(video, canvas, maxSize = 512) {
  const vw = video.videoWidth || maxSize;
  const vh = video.videoHeight || maxSize;
  const scale = Math.min(maxSize / vw, maxSize / vh, 1);
  const tw = Math.max(1, Math.round(vw * scale));
  const th = Math.max(1, Math.round(vh * scale));
  if (canvas.width !== tw) canvas.width = tw;
  if (canvas.height !== th) canvas.height = th;
  canvas.getContext('2d').drawImage(video, 0, 0, tw, th);
  return canvas;
}

function canvasToBlob(canvas, type = 'image/jpeg', quality = 0.68) {
  return new Promise(resolve => canvas.toBlob(resolve, type, quality));
}

function updateWebcamFps(now) {
  state.webcam.framesSinceFps += 1;
  if (!state.webcam.lastFpsAt) state.webcam.lastFpsAt = now;
  const elapsed = now - state.webcam.lastFpsAt;
  if (elapsed >= 1000) {
    const fps = Math.round((state.webcam.framesSinceFps * 1000) / elapsed);
    if (dom.webcamFpsBadge) dom.webcamFpsBadge.textContent = `${fps} FPS`;
    state.webcam.framesSinceFps = 0;
    state.webcam.lastFpsAt = now;
  }
}

function drawRawWebcamFrame(now = performance.now()) {
  if (!state.webcam.active || !dom.webcamVideo || !dom.webcamRawCanvas) return;
  if (!dom.webcamVideo.videoWidth || !dom.webcamVideo.videoHeight) return;
  if (now - state.webcam.lastRawFrameAt < 33) return;
  state.webcam.lastRawFrameAt = now;
  drawVideoToCanvas(dom.webcamVideo, dom.webcamRawCanvas, state.webcam.rawMaxSize || 360);
}

function mediaPipeVisionClass(name) {
  return window[name] || window.vision?.[name] || null;
}

function waitForMediaPipeVision(timeoutMs = 5000) {
  if (mediaPipeVisionClass('FilesetResolver') && mediaPipeVisionClass('FaceLandmarker')) {
    return Promise.resolve(true);
  }
  return new Promise(resolve => {
    let done = false;
    const finish = value => {
      if (done) return;
      done = true;
      window.removeEventListener('mediapipe-vision-ready', onReady);
      clearTimeout(timer);
      resolve(value);
    };
    const onReady = () => finish(true);
    const timer = setTimeout(() => finish(false), timeoutMs);
    window.addEventListener('mediapipe-vision-ready', onReady, { once: true });
  });
}

async function createMediaPipeTask(factory, vision, options) {
  try {
    return await factory.createFromOptions(vision, options);
  } catch (gpuErr) {
    const cpuOptions = {
      ...options,
      baseOptions: {
        ...(options.baseOptions || {}),
        delegate: 'CPU',
      },
    };
    return factory.createFromOptions(vision, cpuOptions);
  }
}

async function initBrowserRealtimeEffects() {
  if (state.webcam.browserEffectsReady) return true;
  if (state.webcam.browserEffectsLoading) return false;
  if (state.webcam.browserEffectsUnavailable) return false;

  await waitForMediaPipeVision();
  const FilesetResolver = mediaPipeVisionClass('FilesetResolver');
  const FaceLandmarker = mediaPipeVisionClass('FaceLandmarker');
  const ImageSegmenter = mediaPipeVisionClass('ImageSegmenter');
  if (!FilesetResolver || !FaceLandmarker) {
    state.webcam.browserEffectsUnavailable = true;
    state.webcam.browserFallbackBackend = true;
    state.webcam.browserEffectsError = 'MediaPipe Tasks Vision bundle is unavailable.';
    return false;
  }

  state.webcam.browserEffectsLoading = true;
  try {
    const vision = await FilesetResolver.forVisionTasks('/static/vendor/mediapipe/wasm');
    state.webcam.visionResolver = vision;
    state.webcam.faceLandmarker = await createMediaPipeTask(FaceLandmarker, vision, {
      baseOptions: {
        modelAssetPath: '/models/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });

    state.webcam.hairSegmenter = await createMediaPipeTask(ImageSegmenter, vision, {
      baseOptions: {
        modelAssetPath: '/models/hair_segmenter.tflite',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
    state.webcam.hairSegmenterAvailable = true;

    state.webcam.browserEffectsReady = true;
    state.webcam.browserEffectsUnavailable = false;
    state.webcam.browserEffectsError = null;
    return true;
  } catch (err) {
    console.error('[FaceWarp] Browser realtime effects unavailable:', err);
    state.webcam.browserEffectsUnavailable = true;
    state.webcam.browserFallbackBackend = true;
    state.webcam.browserEffectsError = err.message || String(err);
    return false;
  } finally {
    state.webcam.browserEffectsLoading = false;
  }
}

function parseHexColor(hex, fallback = [255, 80, 120]) {
  const value = String(hex || '').replace('#', '').trim();
  if (!/^[0-9a-fA-F]{6}$/.test(value)) return fallback;
  return [
    parseInt(value.slice(0, 2), 16),
    parseInt(value.slice(2, 4), 16),
    parseInt(value.slice(4, 6), 16),
  ];
}

function rgbToHslPixel(r, g, b) {
  const rn = Math.max(0, Math.min(255, r)) / 255;
  const gn = Math.max(0, Math.min(255, g)) / 255;
  const bn = Math.max(0, Math.min(255, b)) / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;
  const lightness = (max + min) * 0.5;
  let hue = 0;
  let saturation = 0;

  if (delta > 1e-6) {
    saturation = delta / Math.max(1e-6, 1 - Math.abs(2 * lightness - 1));
    if (max === rn) {
      hue = ((gn - bn) / delta) % 6;
    } else if (max === gn) {
      hue = (bn - rn) / delta + 2;
    } else {
      hue = (rn - gn) / delta + 4;
    }
    hue /= 6;
    if (hue < 0) hue += 1;
  }

  return [hue, Math.max(0, Math.min(1, saturation)), lightness];
}

function hslToRgbPixel(h, s, l) {
  const hue = ((Number(h) || 0) % 1 + 1) % 1;
  const saturation = Math.max(0, Math.min(1, Number(s) || 0));
  const lightness = Math.max(0, Math.min(1, Number(l) || 0));
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const segment = hue * 6;
  const x = chroma * (1 - Math.abs((segment % 2) - 1));
  let r1 = 0;
  let g1 = 0;
  let b1 = 0;

  if (segment < 1) {
    r1 = chroma; g1 = x;
  } else if (segment < 2) {
    r1 = x; g1 = chroma;
  } else if (segment < 3) {
    g1 = chroma; b1 = x;
  } else if (segment < 4) {
    g1 = x; b1 = chroma;
  } else if (segment < 5) {
    r1 = x; b1 = chroma;
  } else {
    r1 = chroma; b1 = x;
  }

  const m = lightness - chroma * 0.5;
  return [
    Math.round((r1 + m) * 255),
    Math.round((g1 + m) * 255),
    Math.round((b1 + m) * 255),
  ];
}

function mixHueUnit(sourceHue, targetHue, amount) {
  let delta = targetHue - sourceHue;
  if (delta > 0.5) delta -= 1;
  if (delta < -0.5) delta += 1;
  return (sourceHue + delta * amount + 1) % 1;
}

function rgba(hex, alpha, fallback) {
  const [r, g, b] = parseHexColor(hex, fallback);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function landmarkPoint(points, idx) {
  return points && points[idx] ? points[idx] : null;
}

function distance(a, b) {
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(a, b) {
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function angleBetween(a, b) {
  if (!a || !b) return 0;
  return Math.atan2(b.y - a.y, b.x - a.x);
}

function drawPolygon(ctx, points, color, alpha = 1, blur = 0) {
  const valid = points.filter(Boolean);
  if (valid.length < 3) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  if (blur > 0) ctx.filter = `blur(${blur}px)`;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(valid[0].x, valid[0].y);
  valid.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawEyeLine(ctx, points, indices, color, width, alpha) {
  const valid = indices.map(idx => landmarkPoint(points, idx)).filter(Boolean);
  if (valid.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(valid[0].x, valid[0].y);
  valid.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
  ctx.stroke();
  ctx.restore();
}

function getWebcamLandmarkPoints(canvas) {
  const raw = state.webcam.lastLandmarks;
  if (!Array.isArray(raw) || !raw.length) return null;
  return raw.map(([x, y]) => ({
    x: Number(x) * canvas.width,
    y: Number(y) * canvas.height,
  }));
}

function smoothWebcamLandmarks(next, latencyMs = 0) {
  if (!Array.isArray(next) || !next.length) return null;
  const prev = state.webcam.smoothedLandmarks;
  const prevVelocity = state.webcam.landmarkVelocity;
  const now = performance.now();
  if (!Array.isArray(prev) || prev.length !== next.length) {
    state.webcam.smoothedLandmarks = next;
    state.webcam.landmarkVelocity = next.map(() => [0, 0]);
    state.webcam.lastSmoothAt = now;
    return next;
  }

  const dt = Math.max(1 / 90, Math.min(0.14, (now - (state.webcam.lastSmoothAt || now)) / 1000 || 1 / 30));
  const lead = Math.max(0, Math.min(0.10, ((Number(latencyMs) || 0) + 16) / 1000));
  const alpha = latencyMs > 90 ? 0.86 : 0.74;
  const beta = latencyMs > 90 ? 0.20 : 0.14;
  const velocityDecay = 0.72;
  const nextVelocity = [];
  const smoothed = next.map((point, idx) => {
    const last = prev[idx] || point;
    const velocity = Array.isArray(prevVelocity?.[idx]) ? prevVelocity[idx] : [0, 0];
    const predicted = [
      Number(last[0]) + Number(velocity[0]) * dt,
      Number(last[1]) + Number(velocity[1]) * dt,
    ];
    const residual = [
      Number(point[0]) - predicted[0],
      Number(point[1]) - predicted[1],
    ];
    const corrected = [
      predicted[0] + residual[0] * alpha,
      predicted[1] + residual[1] * alpha,
    ];
    const updatedVelocity = [
      Number(velocity[0]) * velocityDecay + (residual[0] / dt) * beta,
      Number(velocity[1]) * velocityDecay + (residual[1] / dt) * beta,
    ];
    nextVelocity[idx] = updatedVelocity;
    return [
      Math.max(0, Math.min(1, corrected[0] + updatedVelocity[0] * lead)),
      Math.max(0, Math.min(1, corrected[1] + updatedVelocity[1] * lead)),
    ];
  });
  state.webcam.smoothedLandmarks = smoothed;
  state.webcam.landmarkVelocity = nextVelocity;
  state.webcam.lastSmoothAt = now;
  return smoothed;
}

function currentWebcamEffects() {
  const params = buildRealtimeEffectParams();
  return {
    hair_color: params.hair_color,
    eye_color: params.eye_color,
    makeup: {
      skin_smooth: params.skin_smooth,
      lipstick: params.lipstick,
      beard: params.beard,
      blush: params.blush,
      eyeshadow: params.eyeshadow,
      eyeliner: params.eyeliner,
    },
    accessories: params.accessories,
  };
}

function normalizedMediaPipeLandmarksToCanvas(result, canvas) {
  const face = result?.faceLandmarks?.[0];
  if (!Array.isArray(face) || !face.length) return null;
  const normalized = face.map(point => [
    Math.max(0, Math.min(1, Number(point.x) || 0)),
    Math.max(0, Math.min(1, Number(point.y) || 0)),
    Number(point.z) || 0,
  ]);
  const smoothed = smoothWebcamLandmarks(normalized, 0) || normalized;
  state.webcam.lastLandmarks = smoothed;
  state.webcam.lastLandmarkAt = performance.now();
  return smoothed.map(([x, y]) => ({
    x: x * canvas.width,
    y: y * canvas.height,
  }));
}

function maskToFloatArray(mask, fallbackWidth, fallbackHeight) {
  if (!mask) return null;
  const width = Number(mask.width || mask.canvas?.width || fallbackWidth) || fallbackWidth;
  const height = Number(mask.height || mask.canvas?.height || fallbackHeight) || fallbackHeight;
  let raw = null;
  try {
    if (typeof mask.getAsFloat32Array === 'function') {
      raw = mask.getAsFloat32Array();
    } else if (typeof mask.getAsUint8Array === 'function') {
      raw = mask.getAsUint8Array();
    }
  } catch (err) {
    raw = null;
  }
  if (!raw || !raw.length) return null;

  const data = new Float32Array(raw.length);
  let maxValue = 0;
  for (let i = 0; i < raw.length; i += 1) {
    const v = Number(raw[i]) || 0;
    if (v > maxValue) maxValue = v;
  }
  for (let i = 0; i < raw.length; i += 1) {
    const v = Number(raw[i]) || 0;
    if (maxValue <= 1.01) {
      data[i] = Math.max(0, Math.min(1, v));
    } else {
      data[i] = v > 0 ? 1 : 0;
    }
  }
  try {
    if (typeof mask.close === 'function') mask.close();
  } catch (err) {}
  return { data, width, height };
}

function setBrowserHairMask(result, canvas) {
  const confidenceMasks = Array.isArray(result?.confidenceMasks) ? result.confidenceMasks : [];
  const categoryMask = result?.categoryMask || confidenceMasks[1] || confidenceMasks[0] || null;
  const converted = maskToFloatArray(categoryMask, canvas.width, canvas.height);
  if (!converted) return;
  let sum = 0;
  for (let i = 0; i < converted.data.length; i += 1) {
    sum += converted.data[i] || 0;
  }
  let coverage = converted.data.length ? sum / converted.data.length : 0;
  if (coverage > 0.62) {
    for (let i = 0; i < converted.data.length; i += 1) {
      converted.data[i] = 1 - converted.data[i];
    }
    coverage = 1 - coverage;
  }
  state.webcam.browserHairMask = converted.data;
  state.webcam.browserHairMaskWidth = converted.width;
  state.webcam.browserHairMaskHeight = converted.height;
  state.webcam.browserHairMaskCoverage = coverage;
}

function requestBrowserHairSegmentation(now) {
  const segmenter = state.webcam.hairSegmenter;
  const effects = currentWebcamEffects();
  if (!effects.hair_color?.enabled || !segmenter || !state.webcam.hairSegmenterAvailable) return;
  if (state.webcam.hairSegmentationInFlight) return;
  const intervalMs = 1000 / Math.max(1, Number(state.webcam.hairSegmentFps) || 12);
  if (now - state.webcam.lastHairSegmentAt < intervalMs) return;
  if (!dom.webcamVideo?.videoWidth || !dom.webcamVideo?.videoHeight) return;

  state.webcam.hairSegmentationInFlight = true;
  state.webcam.lastHairSegmentAt = now;
  const watchdog = setTimeout(() => {
    state.webcam.hairSegmentationInFlight = false;
  }, 250);
  const finish = result => {
    try {
      if (dom.webcamCanvas && result) setBrowserHairMask(result, dom.webcamCanvas);
    } finally {
      clearTimeout(watchdog);
      state.webcam.hairSegmentationInFlight = false;
    }
  };

  try {
    const maybe = segmenter.segmentForVideo(dom.webcamVideo, now, finish);
    if (maybe && typeof maybe.then === 'function') {
      maybe.then(finish).catch(err => {
        console.warn('[FaceWarp] Hair segmentation frame failed:', err);
        state.webcam.hairSegmentationInFlight = false;
      });
    } else if (maybe) {
      finish(maybe);
    }
  } catch (err) {
    console.warn('[FaceWarp] Hair segmentation frame failed:', err);
    state.webcam.hairSegmentationInFlight = false;
  }
}

function sampledHairMask(mask, mw, mh, x, y, w, h) {
  if (!mask || !mw || !mh) return 0;
  const gx = (x / Math.max(1, w - 1)) * (mw - 1);
  const gy = (y / Math.max(1, h - 1)) * (mh - 1);
  const x1 = Math.floor(gx);
  const y1 = Math.floor(gy);
  const x2 = Math.min(mw - 1, x1 + 1);
  const y2 = Math.min(mh - 1, y1 + 1);
  const tx = gx - x1;
  const ty = gy - y1;

  const q11 = mask[y1 * mw + x1] || 0;
  const q21 = mask[y1 * mw + x2] || 0;
  const q12 = mask[y2 * mw + x1] || 0;
  const q22 = mask[y2 * mw + x2] || 0;

  const top = q11 * (1 - tx) + q21 * tx;
  const bottom = q12 * (1 - tx) + q22 * tx;
  return top * (1 - ty) + bottom * ty;
}

function applyBrowserHairMaskColor(ctx, params, points = null) {
  if (!params?.enabled) return false;
  const mask = state.webcam.browserHairMask;
  const mw = state.webcam.browserHairMaskWidth;
  const mh = state.webcam.browserHairMaskHeight;
  if (!mask || !mw || !mh) return false;
  if ((Number(state.webcam.browserHairMaskCoverage) || 0) < 0.004) return false;

  const canvas = ctx.canvas;
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = image.data;
  const target = parseHexColor(params.color || '#6f3bb8', [111, 59, 184]);
  const rawIntensity = Number(params.intensity);
  const intensity = Number.isFinite(rawIntensity)
    ? Math.max(0, Math.min(1, rawIntensity))
    : 0.65;
  if (intensity <= 0.001) return false;

  const targetHsl = rgbToHslPixel(target[0], target[1], target[2]);
  const targetLightness = targetHsl[2];
  const isLightTarget = targetLightness >= 135 / 255;
  const lowChromaLight = isLightTarget && targetHsl[1] < 112 / 255;
  const effectiveTargetSaturation = isLightTarget
    ? targetHsl[1] * (lowChromaLight ? 0.68 : 0.88)
    : targetHsl[1];
  let lightnessSum = 0;
  let lightnessWeight = 0;

  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const maskValue = sampledHairMask(mask, mw, mh, x, y, canvas.width, canvas.height);
      if (maskValue <= 0.08) continue;
      const i = (y * canvas.width + x) * 4;
      const maxChannel = Math.max(data[i], data[i + 1], data[i + 2]);
      const minChannel = Math.min(data[i], data[i + 1], data[i + 2]);
      lightnessSum += ((maxChannel + minChannel) / 510) * maskValue;
      lightnessWeight += maskValue;
    }
  }

  const meanLightness = Math.max(
    8 / 255,
    Math.min(247 / 255, lightnessWeight > 1e-6 ? lightnessSum / lightnessWeight : 0.35),
  );
  let toneGamma = 1;
  if (isLightTarget) {
    const lightScore = Math.max(0, Math.min(1, (targetLightness - 135 / 255) / (100 / 255)));
    const toneAmount = intensity * ((lowChromaLight ? 0.72 : 0.82) + 0.08 * lightScore);
    const desired = Math.max(
      10 / 255,
      Math.min(
        (lowChromaLight ? 220 : 228) / 255,
        meanLightness + (targetLightness - meanLightness) * toneAmount,
      ),
    );
    toneGamma = Math.max(0.28, Math.min(1, Math.log(desired) / Math.log(meanLightness)));
  } else if (targetLightness < 65 / 255) {
    const desired = Math.max(
      6 / 255,
      Math.min(245 / 255, meanLightness + (targetLightness - meanLightness) * intensity * 0.32),
    );
    toneGamma = Math.max(1, Math.min(2.2, Math.log(desired) / Math.log(meanLightness)));
  }

  let changed = 0;

  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const m0 = sampledHairMask(mask, mw, mh, x, y, canvas.width, canvas.height);
      if (m0 <= 0.05) continue;

      const i = (y * canvas.width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const m = Math.max(0, Math.min(1, Math.pow((m0 - 0.05) / 0.95, 1.12)));
      if (m <= 0.015) continue;

      const sourceHsl = rgbToHslPixel(r, g, b);
      const hueAmount = m * (
        isLightTarget
          ? (lowChromaLight ? 0.58 + intensity * 0.24 : 0.78 + intensity * 0.18)
          : 0.46 + intensity * 0.54
      );
      const saturationAmount = m * (
        isLightTarget
          ? (lowChromaLight ? 0.46 + intensity * 0.24 : 0.66 + intensity * 0.28)
          : 0.42 + intensity * 0.58
      );
      const newHue = mixHueUnit(sourceHsl[0], targetHsl[0], hueAmount);
      const newSaturation = Math.max(
        0,
        Math.min(
          1,
          sourceHsl[1] * (1 - saturationAmount)
            + effectiveTargetSaturation * saturationAmount,
        ),
      );

      let newLightness = sourceHsl[2];
      if (toneGamma !== 1) {
        const toned = Math.pow(Math.max(0, Math.min(1, sourceHsl[2])), toneGamma);
        const shadowDetail = 0.70
          + 0.30 * Math.max(0, Math.min(1, (sourceHsl[2] - 14 / 255) / (96 / 255)));
        const lumaAlpha = isLightTarget
          ? m * ((lowChromaLight ? 0.64 : 0.70) + intensity * 0.16) * shadowDetail
          : m * (0.42 + intensity * 0.20);
        newLightness = sourceHsl[2] * (1 - lumaAlpha) + toned * lumaAlpha;
      }

      const [tr, tg, tb] = hslToRgbPixel(newHue, newSaturation, newLightness);
      const coreAlpha = isLightTarget
        ? (lowChromaLight ? 0.80 + intensity * 0.10 : 0.84 + intensity * 0.10)
        : 0.87 + intensity * 0.10;
      const alpha = Math.min(0.98, m * coreAlpha);

      data[i] = Math.round(r * (1 - alpha) + tr * alpha);
      data[i + 1] = Math.round(g * (1 - alpha) + tg * alpha);
      data[i + 2] = Math.round(b * (1 - alpha) + tb * alpha);
      changed += 1;
    }
  }

  if (changed < 24) return false;
  ctx.putImageData(image, 0, 0);
  return true;
}

function renderBrowserWebcamEffects(now = performance.now()) {
  if (!state.webcam.active || !dom.webcamVideo || !dom.webcamCanvas) return false;
  if (!dom.webcamVideo.videoWidth || !dom.webcamVideo.videoHeight) return false;

  drawVideoToCanvas(dom.webcamVideo, dom.webcamCanvas, state.webcam.renderMaxSize || 420);
  const ctx = dom.webcamCanvas.getContext('2d');
  const effects = currentWebcamEffects();
  let points = null;

  if (state.webcam.browserEffectsReady && state.webcam.faceLandmarker) {
    try {
      const result = state.webcam.faceLandmarker.detectForVideo(dom.webcamVideo, now);
      points = normalizedMediaPipeLandmarksToCanvas(result, dom.webcamCanvas);
    } catch (err) {
      console.warn('[FaceWarp] Browser landmark frame failed:', err);
    }
  }

  if (effects.hair_color?.enabled) {
    requestBrowserHairSegmentation(now);
    applyBrowserHairMaskColor(ctx, effects.hair_color, points);
  }
  if (points) {
    applyWebcamEyeColor(ctx, points, effects.eye_color);
    applyWebcamMakeup(ctx, points, effects);
    drawWebcamAccessories(ctx, points, effects);
  }

  if (dom.webcamSpinner) dom.webcamSpinner.classList.add('hidden');
  return true;
}

function shouldUseBackendWebcamFallback() {
  return state.webcam.browserFallbackBackend && state.webcam.browserEffectsUnavailable;
}

function applyWebcamHairColor(ctx, points, params) {
  if (!params?.enabled || !Array.isArray(points) || points.length <= 454) return;
  const top = landmarkPoint(points, 10);
  const left = landmarkPoint(points, 234);
  const right = landmarkPoint(points, 454);
  const chin = landmarkPoint(points, 152);
  if (!top || !left || !right || !chin) return;

  const canvas = ctx.canvas;
  const w = canvas.width;
  const h = canvas.height;
  const faceWidth = Math.max(20, distance(left, right));
  const target = parseHexColor(params.color || '#6f3bb8', [111, 59, 184]);
  const intensity = Math.max(0, Math.min(1, Number(params.intensity) || 0.65));

  const x1 = Math.max(0, Math.floor(Math.min(left.x, right.x) - faceWidth * 0.46));
  const x2 = Math.min(w - 1, Math.ceil(Math.max(left.x, right.x) + faceWidth * 0.46));
  const y1 = Math.max(0, Math.floor(top.y - faceWidth * 0.70));
  const y2 = Math.min(h - 1, Math.ceil(top.y + faceWidth * 0.18));
  if (x2 <= x1 || y2 <= y1) return;

  const image = ctx.getImageData(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
  const data = image.data;
  const iw = image.width;
  const ih = image.height;
  const cx = (left.x + right.x) * 0.5;
  const hairCy = top.y - faceWidth * 0.18;
  const hairRx = faceWidth * 0.66;
  const hairRy = faceWidth * 0.55;
  const faceCy = top.y + faceWidth * 0.22;
  const faceRx = faceWidth * 0.40;
  const faceRy = faceWidth * 0.36;
  const targetLum = Math.max(1, target[0] * 0.299 + target[1] * 0.587 + target[2] * 0.114);
  const targetIsBright = targetLum > 90;
  const luma = new Float32Array(iw * ih);
  const satMap = new Float32Array(iw * ih);

  for (let yy = 0; yy < ih; yy += 1) {
    for (let xx = 0; xx < iw; xx += 1) {
      const idx = yy * iw + xx;
      const i = idx * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const maxc = Math.max(r, g, b);
      const minc = Math.min(r, g, b);
      luma[idx] = r * 0.299 + g * 0.587 + b * 0.114;
      satMap[idx] = maxc <= 0 ? 0 : (maxc - minc) / maxc;
    }
  }

  let changed = 0;
  for (let yy = 1; yy < ih - 1; yy += 1) {
    const py = y1 + yy;
    for (let xx = 1; xx < iw - 1; xx += 1) {
      const px = x1 + xx;
      const idx = yy * iw + xx;
      const i = idx * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const lum = luma[idx];
      const sat = satMap[idx];

      const hairEllipse = ((px - cx) ** 2) / (hairRx ** 2) + ((py - hairCy) ** 2) / (hairRy ** 2);
      if (hairEllipse > 1.0) continue;

      const faceEllipse = ((px - cx) ** 2) / (faceRx ** 2) + ((py - faceCy) ** 2) / (faceRy ** 2);
      const skinRgb = r > 58 && g > 40 && b > 28 && r > b + 8 && r >= g * 0.92 && lum > 58;
      const centralForehead = Math.abs(px - cx) < faceWidth * 0.35 && py > top.y - faceWidth * 0.13;
      const lowerFace = faceEllipse < 0.86 && py > top.y + faceWidth * 0.015;
      if ((centralForehead && skinRgb) || lowerFace) continue;

      const gx = Math.abs(luma[idx + 1] - luma[idx - 1]);
      const gy = Math.abs(luma[idx + iw] - luma[idx - iw]);
      const detail = Math.min(1, (gx + gy) / 54);
      const darkScore = Math.max(0, Math.min(1, (148 - lum) / 92));
      const colorScore = sat > 0.12 && lum < 170 ? Math.min(1, sat * 2.4) : 0;
      const hairScore = Math.max(darkScore, colorScore) * (0.72 + detail * 0.38);
      if (hairScore < 0.20) continue;

      const spatial = Math.max(0, Math.min(1, (1.0 - hairEllipse) / 0.42));
      const edgeAlpha = Math.max(0.18, Math.sqrt(spatial));
      const alpha = Math.min(0.94, intensity * hairScore * edgeAlpha * (targetIsBright ? 1.08 : 0.92));
      if (alpha <= 0.025) continue;

      const shade = targetIsBright
        ? Math.max(0.74, Math.min(1.35, 0.72 + lum / 235))
        : Math.max(0.34, Math.min(1.70, (lum + 24) / targetLum));
      const tr = Math.max(0, Math.min(255, target[0] * shade));
      const tg = Math.max(0, Math.min(255, target[1] * shade));
      const tb = Math.max(0, Math.min(255, target[2] * shade));
      data[i] = Math.round(r * (1 - alpha) + tr * alpha);
      data[i + 1] = Math.round(g * (1 - alpha) + tg * alpha);
      data[i + 2] = Math.round(b * (1 - alpha) + tb * alpha);
      changed += 1;
    }
  }

  if (changed > 12) ctx.putImageData(image, x1, y1);
}

function applyWebcamEyeColor(ctx, points, params) {
  if (!params?.enabled || !Array.isArray(points) || points.length < 478) return;
  const rawIntensity = Number(params.intensity);
  const intensity = Number.isFinite(rawIntensity)
    ? Math.max(0, Math.min(1, rawIntensity))
    : 0.45;
  if (intensity <= 0) return;

  const target = parseHexColor(params.color || '#3f7fbf', [63, 127, 191]);
  const targetHsl = rgbToHslPixel(target[0], target[1], target[2]);
  const isGreenTarget = targetHsl[0] >= 35 / 180 && targetHsl[0] <= 85 / 180;
  const strength = 0.51 + intensity * 0.36;
  const eyes = [
    { iris: [468, 469, 470, 471, 472], outer: 33, inner: 133, upper: 159, lower: 145 },
    { iris: [473, 474, 475, 476, 477], outer: 263, inner: 362, upper: 386, lower: 374 },
  ];

  for (const eye of eyes) {
    const iris = eye.iris.map(idx => landmarkPoint(points, idx)).filter(Boolean);
    if (iris.length < 5) continue;

    const outer = landmarkPoint(points, eye.outer);
    const inner = landmarkPoint(points, eye.inner);
    const upper = landmarkPoint(points, eye.upper);
    const lower = landmarkPoint(points, eye.lower);
    if (!outer || !inner || !upper || !lower) continue;

    const eyeCenter = midpoint(outer, inner);
    let center = iris[0];
    const eyeWidth = Math.max(4, distance(outer, inner));
    const eyeHeight = Math.max(2, distance(upper, lower));
    if (!eyeCenter || eyeHeight < eyeWidth * 0.075) continue;
    if (distance(center, eyeCenter) > eyeWidth * 0.24) center = eyeCenter;

    const boundaryDistances = iris
      .slice(1)
      .map(point => distance(point, center))
      .sort((a, b) => a - b);
    const measuredRadius = boundaryDistances[Math.floor(boundaryDistances.length / 2)];
    const radiusX = Math.max(2.0, Math.min(measuredRadius * 1.15, eyeWidth * 0.165));
    const radiusY = Math.max(1.6, Math.min(radiusX, eyeHeight * 0.80));
    const angle = angleBetween(outer, inner);
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const pad = 2;
    const x1 = Math.max(0, Math.floor(center.x - radiusX - pad));
    const y1 = Math.max(0, Math.floor(center.y - radiusY - pad));
    const x2 = Math.min(ctx.canvas.width - 1, Math.ceil(center.x + radiusX + pad));
    const y2 = Math.min(ctx.canvas.height - 1, Math.ceil(center.y + radiusY + pad));
    if (x2 <= x1 || y2 <= y1) continue;

    const image = ctx.getImageData(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
    const data = image.data;
    for (let py = 0; py < image.height; py += 1) {
      for (let px = 0; px < image.width; px += 1) {
        const dx = x1 + px - center.x;
        const dy = y1 + py - center.y;
        const localX = dx * cosA + dy * sinA;
        const localY = -dx * sinA + dy * cosA;
        const radial = Math.sqrt((localX * localX) / (radiusX * radiusX) + (localY * localY) / (radiusY * radiusY));
        if (radial >= 1) continue;

        const i = (py * image.width + px) * 4;
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const sourceHsl = rgbToHslPixel(r, g, b);
        const luma = sourceHsl[2] * 255;

        // Keep the pupil, limbal ring and white catchlights intact. They are
        // what make the recolored iris still look like the original eye.
        const pupilProtect = radial < 0.30 ? Math.max(0, 1 - radial / 0.30) : 0;
        const edgeFeather = Math.max(0, Math.min(1, (1 - radial) / 0.18));
        const highlightProtect = luma > 185 ? Math.min(0.88, (luma - 185) / 70) : 0;
        const darknessProtect = luma < 18 ? Math.min(0.75, (18 - luma) / 18) : 0;
        let alpha = strength * edgeFeather;
        alpha *= 1 - pupilProtect * 0.88;
        alpha *= 1 - highlightProtect;
        alpha *= 1 - darknessProtect;
        if (alpha <= 0.01) continue;

        const visibilityFloor = Math.max(
          targetHsl[2] + (isGreenTarget ? 8 / 255 : 0),
          (80 + intensity * 22) / 255,
        );
        let liftedLightness = sourceHsl[2]
          + Math.max(0, visibilityFloor - sourceHsl[2])
            * intensity
            * 0.40
            * (1 - pupilProtect * 0.92);
        if (isGreenTarget) {
          const middleIris = Math.max(0, 1 - Math.abs(radial - 0.60) / 0.34);
          const limbalRing = Math.max(0, Math.min(1, (radial - 0.80) / 0.20));
          liftedLightness += middleIris * intensity * 0.018;
          liftedLightness -= limbalRing * intensity * 0.022;
        }
        liftedLightness = Math.max(0, Math.min(1, liftedLightness));
        const newHue = mixHueUnit(sourceHsl[0], targetHsl[0], isGreenTarget ? 0.88 : 0.92);
        const targetSaturation = Math.min(
          1,
          targetHsl[1] * (isGreenTarget ? 1.16 : 1.05),
        );
        const newSaturation = Math.max(
          0,
          Math.min(1, sourceHsl[1] * 0.24 + targetSaturation * 0.76),
        );
        const [tr, tg, tb] = hslToRgbPixel(newHue, newSaturation, liftedLightness);
        data[i] = Math.round(r * (1 - alpha) + tr * alpha);
        data[i + 1] = Math.round(g * (1 - alpha) + tg * alpha);
        data[i + 2] = Math.round(b * (1 - alpha) + tb * alpha);
      }
    }
    ctx.putImageData(image, x1, y1);
  }
}

function applyWebcamMakeup(ctx, points, effects) {
  const makeup = effects?.makeup || {};

  if (makeup.skin_smooth?.enabled) {
    const left = landmarkPoint(points, 234);
    const right = landmarkPoint(points, 454);
    const top = landmarkPoint(points, 10);
    const bottom = landmarkPoint(points, 152);
    const center = midpoint(left, right);
    if (center && top && bottom) {
      const intensity = Math.max(0, Math.min(1, Number(makeup.skin_smooth.intensity) || 0.25));
      const rx = Math.max(12, distance(left, right) * 0.58);
      const ry = Math.max(18, distance(top, bottom) * 0.62);
      center.y = (top.y + bottom.y) / 2;
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(center.x, center.y, rx, ry, 0, 0, Math.PI * 2);
      ctx.clip();
      ctx.globalAlpha = 0.18 * intensity;
      ctx.filter = `blur(${1 + intensity * 4}px)`;
      ctx.drawImage(ctx.canvas, 0, 0);
      ctx.filter = 'none';
      ctx.fillStyle = 'rgba(255, 226, 210, 0.08)';
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.restore();
    }
  }

  if (makeup.blush?.enabled) {
    const intensity = Math.max(0, Math.min(1, Number(makeup.blush.intensity) || 0.35));
    const color = parseHexColor(makeup.blush.color, [232, 160, 168]);
    const cheeks = [landmarkPoint(points, 205), landmarkPoint(points, 425)];
    const faceWidth = distance(landmarkPoint(points, 234), landmarkPoint(points, 454));
    cheeks.forEach(center => {
      if (!center) return;
      const radius = Math.max(14, faceWidth * 0.13);
      const gradient = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, radius);
      gradient.addColorStop(0, `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${0.34 * intensity})`);
      gradient.addColorStop(1, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0)`);
      ctx.save();
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  }

  if (makeup.lipstick?.enabled) {
    const intensity = Math.max(0, Math.min(1, Number(makeup.lipstick.intensity) || 0.7));
    const color = rgba(makeup.lipstick.color, 0.42, [176, 0, 32]);
    const upperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291];
    const lowerOuterRightToLeft = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
    const mouthInner = [
      78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
      308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    ];
    const mouthCenter = midpoint(landmarkPoint(points, 61), landmarkPoint(points, 291)) || landmarkPoint(points, 13);
    const outerTop = landmarkPoint(points, 0);
    const outerBottom = landmarkPoint(points, 17);
    const innerTop = landmarkPoint(points, 13);
    const innerBottom = landmarkPoint(points, 14);
    const lipHeight = outerTop && outerBottom ? Math.max(1, Math.abs(outerBottom.y - outerTop.y)) : 1;
    const innerGap = innerTop && innerBottom ? Math.abs(innerBottom.y - innerTop.y) : lipHeight;
    const innerOpenRatio = innerGap / lipHeight;
    const innerScaleX = innerOpenRatio < 0.16 ? 1.0 : 1.04;
    const innerScaleY = innerOpenRatio < 0.16 ? 1.0 : (innerOpenRatio < 0.3 ? 1.08 : 1.16);
    const transformLipPoint = (point, scaleX, scaleY) => {
      if (!point || !mouthCenter) return point;
      return {
        x: mouthCenter.x + (point.x - mouthCenter.x) * scaleX,
        y: mouthCenter.y + (point.y - mouthCenter.y) * scaleY,
      };
    };
    const outerPoints = upperOuter
      .concat(lowerOuterRightToLeft)
      .map(idx => transformLipPoint(landmarkPoint(points, idx), 0.99, 0.98))
      .filter(Boolean);
    const innerPoints = mouthInner
      .map(idx => transformLipPoint(landmarkPoint(points, idx), innerScaleX, innerScaleY))
      .filter(Boolean);
    if (outerPoints.length >= 3) {
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = ctx.canvas.width;
      maskCanvas.height = ctx.canvas.height;
      const maskCtx = maskCanvas.getContext('2d');
      maskCtx.fillStyle = color;
      maskCtx.globalAlpha = intensity;
      maskCtx.beginPath();
      maskCtx.moveTo(outerPoints[0].x, outerPoints[0].y);
      outerPoints.slice(1).forEach(point => maskCtx.lineTo(point.x, point.y));
      maskCtx.closePath();
      maskCtx.fill();
      if (innerPoints.length >= 3) {
        maskCtx.globalCompositeOperation = 'destination-out';
        maskCtx.globalAlpha = 1;
        maskCtx.beginPath();
        maskCtx.moveTo(innerPoints[0].x, innerPoints[0].y);
        innerPoints.slice(1).forEach(point => maskCtx.lineTo(point.x, point.y));
        maskCtx.closePath();
        maskCtx.fill();
      }
      ctx.drawImage(maskCanvas, 0, 0);
    }
  }

  if (makeup.eyeshadow?.enabled) {
    const intensity = Math.max(0, Math.min(1, Number(makeup.eyeshadow.intensity) || 0.4));
    const color = rgba(makeup.eyeshadow.color, 0.32 * intensity, [140, 122, 107]);
    [
      [landmarkPoint(points, 33), landmarkPoint(points, 133), landmarkPoint(points, 159), landmarkPoint(points, 145)],
      [landmarkPoint(points, 362), landmarkPoint(points, 263), landmarkPoint(points, 386), landmarkPoint(points, 374)],
    ].forEach(([a, b, upper, lower]) => {
      const center = midpoint(a, b);
      if (!center || !upper || !lower) return;
      const rx = Math.max(8, distance(a, b) * 0.64);
      const ry = Math.max(5, distance(upper, lower) * 1.8);
      ctx.save();
      ctx.translate(center.x, center.y - ry * 0.35);
      ctx.rotate(angleBetween(a, b));
      ctx.fillStyle = color;
      ctx.filter = 'blur(1.5px)';
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, Math.PI, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  }

  if (makeup.eyeliner?.enabled) {
    const intensity = Math.max(0, Math.min(1, Number(makeup.eyeliner.intensity) || 0.5));
    const linerRgb = parseHexColor(makeup.eyeliner.color, [8, 8, 8]).map(v => Math.round(v * 0.72));
    const color = `rgba(${linerRgb[0]}, ${linerRgb[1]}, ${linerRgb[2]}, ${0.58 * (0.65 + intensity * 0.35)})`;
    const faceWidth = distance(landmarkPoint(points, 234), landmarkPoint(points, 454));
    const width = Math.max(1.2, faceWidth * (0.006 + 0.004 * intensity));
    const drawLiner = (upperIdx, lowerIdx, outerIdx, innerIdx, topIdx, dir) => {
      const outer = landmarkPoint(points, outerIdx);
      const inner = landmarkPoint(points, innerIdx);
      const top = landmarkPoint(points, topIdx);
      if (!outer || !inner || !top) return;
      const eyeWidth = distance(outer, inner);
      const lift = Math.max(1, eyeWidth * 0.025);
      const upperPoints = upperIdx
        .map(idx => landmarkPoint(points, idx))
        .filter(Boolean)
        .map(point => ({ x: point.x, y: point.y - lift }));
      if (upperPoints.length >= 2) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(upperPoints[0].x, upperPoints[0].y);
        upperPoints.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
        ctx.stroke();
        ctx.restore();
      }

      if (intensity > 0.78) {
        drawEyeLine(ctx, points, lowerIdx, color, Math.max(1, width - 0.5), 0.35);
      }

      const wingLen = eyeWidth * (0.055 + 0.07 * intensity);
      const wingUp = Math.max(1.5, eyeWidth * (0.025 + 0.035 * intensity));
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1, width - 0.5);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(outer.x, outer.y - lift);
      ctx.lineTo(outer.x + dir * wingLen, top.y - wingUp);
      ctx.stroke();
      ctx.restore();
    };
    drawLiner(
      [33, 246, 161, 160, 159, 158, 157, 173, 133],
      [33, 7, 163, 144, 145, 153, 154, 155, 133],
      33,
      133,
      159,
      -1,
    );
    drawLiner(
      [362, 398, 384, 385, 386, 387, 388, 466, 263],
      [362, 382, 381, 380, 374, 373, 390, 249, 263],
      263,
      362,
      386,
      1,
    );
  }
}

function selectedAssetRecord(category, assetId) {
  const assets = state.assetManifest?.categories?.[category] || [];
  return assets.find(asset => asset.id === assetId) || null;
}

function webcamAssetImage(asset) {
  const path = asset?.path ? `/${asset.path}` : '';
  if (!path) return null;
  if (state.webcam.assetImageCache.has(path)) return state.webcam.assetImageCache.get(path);
  const img = new Image();
  img.decoding = 'async';
  img.src = path;
  state.webcam.assetImageCache.set(path, img);
  return img;
}

function drawRotatedImage(ctx, img, cx, cy, width, height, angle = 0, alpha = 1) {
  if (!img || !img.complete || !img.naturalWidth) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.drawImage(img, -width / 2, -height / 2, width, height);
  ctx.restore();
}

  function clearWebcamProcessedFrame() {
    state.webcam.processedFrameImage = null;
    state.webcam.processedFrameInFlight = false;
    state.webcam.backendConsecutiveErrors = 0;
    if (state.webcam.processedFrameObjectUrl) {
      URL.revokeObjectURL(state.webcam.processedFrameObjectUrl);
      state.webcam.processedFrameObjectUrl = null;
    }
  }

  function drawWebcamProcessedFrame() {
    const img = state.webcam.processedFrameImage;
    if (!img || !img.complete || !img.naturalWidth || !dom.webcamCanvas) return false;

    if (dom.webcamCanvas.width !== img.naturalWidth) dom.webcamCanvas.width = img.naturalWidth;
    if (dom.webcamCanvas.height !== img.naturalHeight) dom.webcamCanvas.height = img.naturalHeight;
    const ctx = dom.webcamCanvas.getContext('2d');
    ctx.clearRect(0, 0, dom.webcamCanvas.width, dom.webcamCanvas.height);
    ctx.drawImage(img, 0, 0, dom.webcamCanvas.width, dom.webcamCanvas.height);
    return true;
  }

  async function pollWebcamProcessedFrame() {
    if (!state.webcam.active || state.webcam.processedFrameInFlight || !dom.webcamVideo) return;
    if (!dom.webcamVideo.videoWidth || !dom.webcamVideo.videoHeight) return;

    const params = buildRealtimeEffectParams();

    state.webcam.processedFrameInFlight = true;
    try {
      const startedAt = performance.now();
      const sourceCanvas = state.webcam.processedCanvas || document.createElement('canvas');
      state.webcam.processedCanvas = sourceCanvas;
      drawVideoToCanvas(dom.webcamVideo, sourceCanvas, state.webcam.effectMaxSize || 360);
      const blob = await canvasToBlob(sourceCanvas, 'image/jpeg', 0.65);
      if (!blob) throw new Error('Could not capture webcam frame.');

      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');
      if (state.webcam.sessionId) {
        formData.append('session_id', state.webcam.sessionId);
      }
      formData.append('response_format', 'image/jpeg');
      formData.append('params_json', JSON.stringify(params));

      const res = await fetch('/api/realtime/frame', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(text || `HTTP ${res.status}`);
      }

      const sessionId = res.headers.get('X-Realtime-Session-Id');
      if (sessionId) state.webcam.sessionId = sessionId;

      const frameBlob = await res.blob();
      if (!frameBlob.size) return;
      if (state.webcam.processedFrameObjectUrl) {
        URL.revokeObjectURL(state.webcam.processedFrameObjectUrl);
      }
      const objectUrl = URL.createObjectURL(frameBlob);
      state.webcam.processedFrameObjectUrl = objectUrl;

      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = objectUrl;
      });
      state.webcam.processedFrameImage = img;
      state.webcam.backendConsecutiveErrors = 0;

      const latencyMs = performance.now() - startedAt;
      if (latencyMs > 180) {
        state.webcam.backendFrameFps = 12;
      } else if (latencyMs > 120) {
        state.webcam.backendFrameFps = 18;
      } else if (latencyMs < 55) {
        state.webcam.backendFrameFps = 24;
      } else {
        state.webcam.backendFrameFps = 20;
      }
    } catch (err) {
      state.webcam.backendConsecutiveErrors += 1;
      if (state.webcam.backendConsecutiveErrors <= 3) {
        console.error('[FaceWarp] Webcam processed frame error:', err);
      }
      if (state.webcam.backendConsecutiveErrors >= 6) {
        clearWebcamProcessedFrame();
      }
    } finally {
      state.webcam.processedFrameInFlight = false;
    }
  }

function drawWebcamAccessories(ctx, points, effects) {
  const items = effects?.accessories?.items || [];
  if (!Array.isArray(items) || !items.length) return;

  const leftEye = landmarkPoint(points, 33);
  const rightEye = landmarkPoint(points, 263);
  const leftInner = landmarkPoint(points, 133) || leftEye;
  const rightInner = landmarkPoint(points, 362) || rightEye;
  const faceLeft = landmarkPoint(points, 234);
  const faceRight = landmarkPoint(points, 454);
  const chin = landmarkPoint(points, 152);
  const nose = landmarkPoint(points, 1);
  const faceWidth = Math.max(1, distance(faceLeft, faceRight) || distance(leftEye, rightEye) * 2.4);
  const roll = angleBetween(leftEye, rightEye);

  items.forEach(item => {
    const category = item.category || item.type;
    if (!['glasses', 'earrings', 'necklaces', 'hair_clips'].includes(category)) return;
    const asset = selectedAssetRecord(category, item.asset_id);
    const img = webcamAssetImage(asset);
    const scale = Math.max(0.05, Number(item.scale) || Number(asset?.default_scale) || 1);
    const offsetX = Number(item.offset_x) || 0;
    const offsetY = Number(item.offset_y) || 0;
    const alpha = Math.max(0.1, Math.min(1, Number(item.alpha ?? asset?.default_alpha ?? 0.96)));

    if (category === 'glasses') {
      const center = midpoint(leftInner, rightInner) || midpoint(leftEye, rightEye);
      if (!center) return;
      const width = faceWidth * 0.92 * scale;
      const ratio = img?.naturalWidth ? img.naturalHeight / img.naturalWidth : 0.34;
      drawRotatedImage(
        ctx,
        img,
        center.x + faceWidth * offsetX,
        center.y + faceWidth * (0.03 + offsetY),
        width,
        width * ratio,
        roll,
        alpha
      );
    } else if (category === 'earrings') {
      const anchors = [landmarkPoint(points, 177) || faceLeft, landmarkPoint(points, 401) || faceRight];
      anchors.forEach((anchor, sideIdx) => {
        if (!anchor) return;
        const width = faceWidth * 0.18 * scale;
        const ratio = img?.naturalWidth ? img.naturalHeight / img.naturalWidth : 1.35;
        const side = sideIdx === 0 ? -1 : 1;
        drawRotatedImage(
          ctx,
          img,
          anchor.x + side * faceWidth * (0.04 + offsetX),
          anchor.y + faceWidth * (0.10 + offsetY),
          width,
          width * ratio,
          roll * 0.35,
          alpha
        );
      });
    } else if (category === 'necklaces') {
      if (!chin || !nose) return;
      const centerX = ((faceLeft?.x || chin.x) + (faceRight?.x || chin.x)) / 2 + faceWidth * offsetX;
      const centerY = chin.y + faceWidth * (0.42 + offsetY);
      const width = faceWidth * 1.18 * scale;
      const ratio = img?.naturalWidth ? img.naturalHeight / img.naturalWidth : 0.38;
      drawRotatedImage(ctx, img, centerX, centerY, width, width * ratio, roll * 0.2, alpha);
    } else if (category === 'hair_clips') {
      const temple = landmarkPoint(points, 127) || faceLeft;
      if (!temple) return;
      const width = faceWidth * 0.28 * scale;
      const ratio = img?.naturalWidth ? img.naturalHeight / img.naturalWidth : 0.55;
      drawRotatedImage(
        ctx,
        img,
        temple.x - faceWidth * (0.04 - offsetX),
        temple.y - faceWidth * (0.18 - offsetY),
        width,
        width * ratio,
        roll - 0.22,
        alpha
      );
    }
  });
}

  function renderWebcamArFrame(now = performance.now()) {
    if (!state.webcam.active || !dom.webcamVideo || !dom.webcamCanvas) return;
    if (!dom.webcamVideo.videoWidth || !dom.webcamVideo.videoHeight) return;

    if (!renderBrowserWebcamEffects(now) && !drawWebcamProcessedFrame()) {
      drawVideoToCanvas(dom.webcamVideo, dom.webcamCanvas, state.webcam.renderMaxSize || 420);
    }

    if (dom.webcamSpinner) dom.webcamSpinner.classList.add('hidden');
    updateWebcamFps(now);
  }

function webcamLoop(now = performance.now()) {
  if (!state.webcam.active) return;
  const renderIntervalMs = 1000 / Math.max(1, Number(state.webcam.targetFps) || 30);
  if (now - state.webcam.lastFrameAt >= renderIntervalMs) {
    state.webcam.lastFrameAt = now;
    drawRawWebcamFrame(now);
    renderWebcamArFrame(now);
  }
  const backendIntervalMs = 1000 / Math.max(1, Number(state.webcam.backendFrameFps) || 12);
  if (
    shouldUseBackendWebcamFallback() &&
    !state.webcam.processedFrameInFlight &&
    now - state.webcam.lastProcessedFrameAt >= backendIntervalMs
  ) {
    state.webcam.lastProcessedFrameAt = now;
    pollWebcamProcessedFrame();
  }
  state.webcam.rafId = requestAnimationFrame(webcamLoop);
}

async function startWebcamMode() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast('Camera API is not available in this browser.', 'error');
    return;
  }
  if (state.webcam.active) return;

  try {
    setStatus('Opening camera...', 'processing');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 640 },
        frameRate: { ideal: 30, max: 30 },
        facingMode: 'user',
      },
      audio: false,
    });

    state.webcam.active = true;
    state.webcam.stream = stream;
    state.webcam.sessionId = null;
    state.webcam.lastFrameAt = 0;
    state.webcam.lastTrackAt = 0;
    state.webcam.lastProcessedFrameAt = 0;
    state.webcam.lastRawFrameAt = 0;
    state.webcam.lastFpsAt = 0;
    state.webcam.framesSinceFps = 0;
    state.webcam.targetFps = 30;
    state.webcam.trackingFps = 24;
    state.webcam.backendFrameFps = 12;
    state.webcam.backendConsecutiveErrors = 0;
    state.webcam.browserFallbackBackend = false;
    state.webcam.browserEffectsUnavailable = false;
    state.webcam.browserEffectsError = null;
    state.webcam.hairSegmentationInFlight = false;
    state.webcam.lastHairSegmentAt = 0;
    state.webcam.browserHairMask = null;
    state.webcam.browserHairMaskWidth = 0;
    state.webcam.browserHairMaskHeight = 0;
    state.webcam.browserHairMaskCoverage = 0;
    state.webcam.lastLandmarks = null;
    state.webcam.smoothedLandmarks = null;
    state.webcam.landmarkVelocity = null;
    state.webcam.lastSmoothAt = 0;
    state.webcam.lastLandmarkAt = 0;
    clearWebcamProcessedFrame();
    state.webcam.sourceCanvas = state.webcam.sourceCanvas || document.createElement('canvas');

    if (dom.webcamVideo) {
      dom.webcamVideo.srcObject = stream;
      await dom.webcamVideo.play();
    }
    setWebcamVisible(true);
    setStatus('Webcam Live', 'processing');
    initBrowserRealtimeEffects().then(ok => {
      if (!ok) {
        showToast('Browser realtime effects unavailable; using backend fallback.', 'warning', 4000);
      }
    });
    webcamLoop();
  } catch (err) {
    console.error('[FaceWarp] Webcam start error:', err);
    setStatus('Camera Error', 'error');
    showToast(err.message || 'Could not open webcam.', 'error', 5000);
    stopWebcamMode();
  }
}

  function stopWebcamMode() {
    const realtimeSessionId = state.webcam.sessionId;
    state.webcam.active = false;
  if (state.webcam.rafId) {
    cancelAnimationFrame(state.webcam.rafId);
    state.webcam.rafId = null;
  }
  if (state.webcam.stream) {
    state.webcam.stream.getTracks().forEach(track => track.stop());
  }
  state.webcam.stream = null;
  state.webcam.sessionId = null;
    state.webcam.inFlight = false;
    state.webcam.trackingInFlight = false;
    state.webcam.processedFrameInFlight = false;
    state.webcam.hairSegmentationInFlight = false;
    state.webcam.browserFallbackBackend = false;
    state.webcam.browserHairMask = null;
    state.webcam.browserHairMaskWidth = 0;
    state.webcam.browserHairMaskHeight = 0;
    state.webcam.browserHairMaskCoverage = 0;
    state.webcam.lastLandmarks = null;
    state.webcam.smoothedLandmarks = null;
    state.webcam.landmarkVelocity = null;
    state.webcam.lastSmoothAt = 0;
    clearWebcamProcessedFrame();
  if (state.webcam.lastObjectUrl) {
    URL.revokeObjectURL(state.webcam.lastObjectUrl);
    state.webcam.lastObjectUrl = null;
    }
    if (dom.webcamVideo) dom.webcamVideo.srcObject = null;
    if (dom.webcamFpsBadge) dom.webcamFpsBadge.textContent = '0 FPS';
    setWebcamVisible(false);
    setStatus('Ready', 'idle');
    if (realtimeSessionId) {
      fetch(`/api/realtime/${encodeURIComponent(realtimeSessionId)}`, {
        method: 'DELETE',
      }).catch(() => {});
    }
  }

async function captureWebcamPhoto() {
  if (!dom.webcamRawCanvas || !state.webcam.active) return;
  drawVideoToCanvas(dom.webcamVideo, dom.webcamRawCanvas, state.webcam.rawMaxSize || 480);
  const blob = await canvasToBlob(dom.webcamRawCanvas, 'image/png');
  if (!blob) {
    showToast('Could not capture current webcam frame.', 'error');
    return;
  }
  const file = new File([blob], `webcam-capture-${Date.now()}.png`, { type: 'image/png' });
  stopWebcamMode();
  handleFile(file);
}

dom.btnProcess.addEventListener('click', async () => {
  if (!state.file || !state.sessionId || !state.imageId || state.processing) return;
  if (state.mode === 'ai_expression') {
    await processAiExpression();
    return;
  }
  if (state.mode === 'virtual_tryon') {
    await processVirtualTryOn();
    return;
  }
  await processImage();
});

if (dom.btnAiExpression) {
  dom.btnAiExpression.addEventListener('click', async () => {
    if (!state.file || !state.sessionId || !state.imageId || state.processing) return;
    await processAiExpression();
  });
}

if (dom.btnWebcam) {
  dom.btnWebcam.addEventListener('click', startWebcamMode);
}

if (dom.btnWebcamClose) {
  dom.btnWebcamClose.addEventListener('click', stopWebcamMode);
}

if (dom.btnWebcamCapture) {
  dom.btnWebcamCapture.addEventListener('click', captureWebcamPhoto);
}

async function processVirtualTryOn() {
  if (!state.sessionId || !state.imageId) {
    showToast('Upload the person image first (wait for upload to finish).', 'warning');
    return;
  }
  const storeItem = selectedStoreItem();
  const useStoreGarment = storeItem?.pipeline === 'virtual_tryon';
  if (!useStoreGarment && !state.tryon.garmentFile) {
    showToast('Select a store garment or upload a garment image first.', 'warning');
    return;
  }

  state.processing = true;
  setProcessingUI(true);
  setStatus('Running virtual try-on...', 'processing');

  const startTime = performance.now();
  try {
    const formData = new FormData();
    formData.append('session_id', state.sessionId);
    formData.append('image_id', state.imageId);
    if (useStoreGarment) {
      formData.append('item_id', storeItem.id);
    } else {
      formData.append('cloth', state.tryon.garmentFile);
    }
    formData.append('model_type', state.tryon.modelType || 'dc');
    formData.append('category', state.tryon.category || 'upperbody');
    formData.append('sample', String(state.tryon.sample || 1));
    formData.append('steps', String(state.tryon.steps || 20));
    formData.append('scale', String(state.tryon.scale || 2.0));
    formData.append('seed', String(state.tryon.seed ?? -1));

    const response = await fetch(useStoreGarment ? '/api/store/tryon' : '/api/tryon/process', {
      method: 'POST',
      body: formData,
    });
    const elapsed = Math.round(performance.now() - startTime);
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const msg =
        data?.error?.message ??
        (typeof data?.error === 'string' ? data.error : null) ??
        (typeof data?.detail === 'string' ? data.detail : data?.detail?.[0]?.msg) ??
        `HTTP ${response.status}`;
      throw new Error(msg);
    }

    if (!data.success) {
      const reason = data.error || data.status?.reason || 'virtual_tryon_unavailable';
      const message = tryonReasonLabel(reason);
      setTryonStatusNote(message, 'warning');
      setStatus('Try-On Not Ready', 'error');
      showToast(message, 'warning', 7000);
      setProcessingUI(false);
      return;
    }

    const resultData = {
      ...data,
      elapsed,
      mode: 'virtual_tryon',
      paths: {
        preprocessed_path: data.output_paths?.[0] || null,
      },
      metrics: data.metrics || null,
    };
    state.resultData = resultData;
    state.lastEffectsMeta = [{
      effect: 'virtual_tryon',
      provider: data.provider || 'OOTDiffusion',
      applied: true,
      fallback_used: data.fallback_used === true,
      error: data.error || null,
      quality_warning: data.quality_warning || null,
      model_type: data.model_type,
      category: data.category,
      store_item: data.store_item || null,
    }];
    renderResult(resultData, elapsed);
    if (data.fallback_used === true) {
      setTryonStatusNote(data.quality_warning || 'CPU preview fallback generated a test result.', 'warning');
    } else {
      setTryonStatusNote('Virtual try-on completed.', 'success');
    }
  } catch (err) {
    console.error('[FaceWarp] Virtual try-on error:', err);
    setStatus('Error', 'error');
    showToast(err.message || 'Virtual try-on failed.', 'error', 5000);
    setProcessingUI(false);
  } finally {
    state.processing = false;
  }
}

async function processImage(options = {}) {
  if (!state.sessionId || !state.imageId) {
    showToast('Upload the image first (wait for upload to finish).', 'warning');
    return;
  }
  state.processing = true;
  setProcessingUI(true);
  setStatus('Processing…', 'processing');

  const startTime = performance.now();

  try {
    const params = {
      smile_intensity: state.smileIntensity,
      eyebrow_height:  state.eyebrowHeight,
      lip_intensity:   state.lipIntensity,
      face_slimming:   state.faceSlimming,
      aging_intensity: state.agingIntensity,
      target_age:      state.agingAlgorithm === 'ai' ? state.agingIntensity : undefined,
      item_type:       state.accessoryItem,  // used when mode === 'accessory'
    };

    const effects = state.mode === 'accessory'
      ? buildEditorEffectsPayload()
      : undefined;

    const payload = {
      aging_algorithm: state.agingAlgorithm,
      session_id:      state.sessionId,
      image_id:        state.imageId,
      mode:            state.mode,
      params,
      options: {
        target_size:   512,
        normalize_rgb: true,
        grayscale:     false,
        debug:         state.mode === 'accessory',
      },
    };

    if (effects) {
      payload.effects = effects;
      state.lastEditorPayload = payload;
      updateEditorDebug();
    }

    const response = await fetch('/api/process', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const elapsed = Math.round(performance.now() - startTime);

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({ error: 'Server error' }));
      // Backend returns { error: { code, message } } or FastAPI { detail: ... }
      const msg =
        errBody?.error?.message ??
        (typeof errBody?.error === 'string' ? errBody.error : null) ??
        (typeof errBody?.detail === 'string' ? errBody.detail : errBody?.detail?.[0]?.msg) ??
        `HTTP ${response.status}`;
      throw new Error(msg);
    }

    const data = await response.json();

    const hasLandmarks = data.landmark_detection?.points?.length > 0;
    if (!hasLandmarks) {
      showToast('No face detected! Please upload a clear frontal image.', 'error', 5000);
      setStatus('Face Not Found', 'error');
      setProcessingUI(false);
      state.processing = false;
      return
    }

    state.resultData = { ...data, elapsed };
    state.lastEffectsMeta = data.effects_meta || null;
    renderResult(data, elapsed);
    updateEditorDebug();

  } catch (err) {
    console.error('[FaceWarp] Process error:', err);
    setStatus('Error', 'error');
    showToast(err.message || 'Processing failed. Is the backend running?', 'error', 5000);
    setProcessingUI(false);
  } finally {
    state.processing = false;
  }
}

async function processAiExpression() {
  if (!state.sessionId || !state.imageId) {
    showToast('Upload the image first (wait for upload to finish).', 'warning');
    return;
  }

  state.processing = true;
  setProcessingUI(true);
  setStatus('Testing AI expression...', 'processing');

  const startTime = performance.now();
  const selectedTemplate = state.aiDrivingTemplate || 'direct_smile';
  const useDrivingTemplate = state.aiUseDrivingTemplate && selectedTemplate !== 'direct_smile';
  const selectedPreset = state.aiExpressionPreset || 'smile';
  const internalPreset = aiScoringPresetFor(selectedPreset, selectedTemplate, useDrivingTemplate);
  const safeAiIntensity = selectedPreset === 'neutral'
    ? 0.0
    : Math.min(Number(state.aiExpressionIntensity) || 0, 1.0);
  const payload = {
    session_id: state.sessionId,
    image_id: state.imageId,
    mode: 'ai_expression',
    params: {
      expression_preset: selectedPreset,
      expression_intensity: safeAiIntensity,
      use_liveportrait: true,
      fallback_to_legacy: !useDrivingTemplate,
      use_driving_template: useDrivingTemplate,
      driving_template: useDrivingTemplate ? selectedTemplate : 'direct_smile',
      internal_preset: internalPreset,
      candidate_frame_override: state.aiCandidateFrameOverride || 'auto',
    },
  };

  console.info('[FaceWarp] AI expression request payload:', payload);

  try {
    const response = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const elapsed = Math.round(performance.now() - startTime);

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({ error: 'Server error' }));
      const msg =
        errBody?.error?.message ??
        (typeof errBody?.error === 'string' ? errBody.error : null) ??
        (typeof errBody?.detail === 'string' ? errBody.detail : errBody?.detail?.[0]?.msg) ??
        `HTTP ${response.status}`;
      throw new Error(msg);
    }

    const data = await response.json();
    state.resultData = { ...data, elapsed };
    renderResult(data, elapsed);
    renderAiExpressionDebug(data.ai_expression ?? null);

    if (data.ai_expression?.fallback_used === true) {
      showToast('AI expression bridge is not implemented yet; legacy fallback was used.', 'warning', 6000);
    }
  } catch (err) {
    console.error('[FaceWarp] AI expression error:', err);
    setStatus('Error', 'error');
    showToast(err.message || 'AI expression test failed. Is the backend running?', 'error', 5000);
    setProcessingUI(false);
  } finally {
    state.processing = false;
    if (state.autoApplyQueued) {
      state.autoApplyQueued = false;
      scheduleAutoApply('queued editor change');
    }
  }
}

/* ═══════════════════════════════════════════════
   Render result from API
═══════════════════════════════════════════════ */

function renderAiExpressionDebug(meta) {
  if (!dom.aiExpressionDebug) return;

  if (!meta) {
    dom.aiExpressionDebug.classList.add('hidden');
    dom.aiDebugProvider.textContent = '-';
    dom.aiDebugMode.textContent = '-';
    dom.aiDebugTemplate.textContent = '-';
    dom.aiDebugPreset.textContent = '-';
    dom.aiDebugFrameCount.textContent = '-';
    dom.aiDebugSelectedFrame.textContent = '-';
    dom.aiDebugExpressionScore.textContent = '-';
    if (dom.aiDebugTopFrames) dom.aiDebugTopFrames.textContent = '-';
    if (dom.aiDebugScoring) dom.aiDebugScoring.textContent = '-';
    if (dom.aiDebugCandidateDir) dom.aiDebugCandidateDir.textContent = '-';
    if (dom.aiDebugBrowPx) dom.aiDebugBrowPx.textContent = '-';
    if (dom.aiDebugEyePx) dom.aiDebugEyePx.textContent = '-';
    if (dom.aiDebugIrisPx) dom.aiDebugIrisPx.textContent = '-';
    if (dom.aiDebugLiftPx) dom.aiDebugLiftPx.textContent = '-';
    dom.aiDebugFiles.textContent = '-';
    dom.aiDebugBridge.textContent = '-';
    dom.aiDebugFallback.textContent = '-';
    dom.aiDebugError.textContent = '-';
    dom.aiExpressionFallbackNote.classList.add('hidden');
    return;
  }

  dom.aiExpressionDebug.classList.remove('hidden');
  const liveportraitMeta = meta.liveportrait || {};
  dom.aiDebugProvider.textContent = String(meta.provider ?? '-');
  dom.aiDebugMode.textContent = String(liveportraitMeta.mode ?? meta.mode ?? '-');
  dom.aiDebugTemplate.textContent = String(liveportraitMeta.used_driving_template ?? meta.used_driving_template ?? '-');
  dom.aiDebugPreset.textContent = String(liveportraitMeta.preset ?? meta.requested_preset ?? meta.preset ?? '-');
  dom.aiDebugFrameCount.textContent = String(liveportraitMeta.frame_count ?? meta.frame_count ?? '-');
  dom.aiDebugSelectedFrame.textContent = String(liveportraitMeta.selected_frame_index ?? liveportraitMeta.frame_index ?? meta.selected_frame_index ?? meta.frame_index ?? '-');
  const expressionScore = liveportraitMeta.selected_expression_score ?? meta.selected_expression_score;
  dom.aiDebugExpressionScore.textContent =
    typeof expressionScore === 'number' ? expressionScore.toFixed(3) : String(expressionScore ?? '-');
  const topFrames = liveportraitMeta.top_frame_indices ?? meta.top_frame_indices;
  if (dom.aiDebugTopFrames) {
    dom.aiDebugTopFrames.textContent = Array.isArray(topFrames)
      ? topFrames.join(', ')
      : String(topFrames ?? '-');
  }
  if (dom.aiDebugScoring) {
    dom.aiDebugScoring.textContent = String(
      liveportraitMeta.scoring_method ??
      meta.scoring_method ??
      liveportraitMeta.representative_frame_strategy ??
      '-'
    );
  }
  if (dom.aiDebugCandidateDir) {
    dom.aiDebugCandidateDir.textContent = String(
      liveportraitMeta.candidate_dir ??
      meta.candidate_dir ??
      '-'
    );
  }
  const eyebrowMeta = meta.eyebrow_raise || liveportraitMeta.eyebrow_raise || {};
  const fmtPx = value => typeof value === 'number' ? value.toFixed(2) : String(value ?? '-');
  if (dom.aiDebugBrowPx) {
    dom.aiDebugBrowPx.textContent = fmtPx(
      eyebrowMeta.brow_vertical_shift_px ??
      eyebrowMeta.eyebrow_vertical_displacement_px
    );
  }
  if (dom.aiDebugEyePx) {
    dom.aiDebugEyePx.textContent = fmtPx(
      eyebrowMeta.eye_corner_shift_px ??
      eyebrowMeta.eye_corner_max_displacement_px
    );
  }
  if (dom.aiDebugIrisPx) {
    dom.aiDebugIrisPx.textContent = fmtPx(
      eyebrowMeta.iris_shift_px ??
      eyebrowMeta.iris_max_displacement_px
    );
  }
  if (dom.aiDebugLiftPx) {
    dom.aiDebugLiftPx.textContent = fmtPx(eyebrowMeta.lift_px);
  }
  dom.aiDebugFiles.textContent = String(meta.files_available ?? '-');
  dom.aiDebugBridge.textContent = String(meta.inference_bridge_implemented ?? '-');
  dom.aiDebugFallback.textContent = String(meta.fallback_used ?? '-');
  dom.aiDebugError.textContent = meta.error ? String(meta.error) : '-';
  dom.aiExpressionFallbackNote.classList.toggle('hidden', meta.fallback_used !== true);
}

function renderResult(data, elapsed) {
  renderAiExpressionDebug(data.ai_expression ?? null);

  // ── Result image ──────────────────────────────
  const src = data.result_image
    ? `data:image/png;base64,${data.result_image}`
    : data.paths?.preprocessed_path
      ? `/${data.paths.preprocessed_path}?t=${Date.now()}`
      : '';

  if (src) {
    dom.imgResult.src = src;
    dom.imgResult.classList.remove('hidden');
    dom.resultPlaceholder.classList.add('hidden');
    dom.labelResult.textContent = `Warped · ${elapsed} ms`;
  }

  // ── FFT — Original Magnitude ──────────────────
  if (data.paths?.fft_orig_path) {
    showFFTImage(
      dom.imgFftOrig,
      dom.fftOrigPholder,
      '/' + data.paths.fft_orig_path + '?t=' + Date.now(),
      () => runFftWhenImageReady(dom.imgOriginal, dom.imgFftOrig, dom.fftOrigPholder)
    );
  } else if (dom.imgOriginal.naturalWidth) {
    runFftWhenImageReady(dom.imgOriginal, dom.imgFftOrig, dom.fftOrigPholder);
  }

  // ── FFT — Original Phase ──────────────────────
  if (data.paths?.fft_phase_orig_path) {
    showFFTImage(
      dom.imgFftPhaseOrig,
      dom.fftPhaseOrigPholder,
      '/' + data.paths.fft_phase_orig_path + '?t=' + Date.now(),
      () => runFftWhenImageReady(dom.imgOriginal, dom.imgFftPhaseOrig, dom.fftPhaseOrigPholder, true)
    );
  } else if (dom.imgOriginal.naturalWidth) {
    runFftWhenImageReady(dom.imgOriginal, dom.imgFftPhaseOrig, dom.fftPhaseOrigPholder, true);
  }

  // ── FFT — Warped Magnitude ────────────────────
  if (data.paths?.fft_proc_path) {
    showFFTImage(
      dom.imgFftResult,
      dom.fftResultPholder,
      '/' + data.paths.fft_proc_path + '?t=' + Date.now(),
      () => runFftWhenImageReady(dom.imgResult, dom.imgFftResult, dom.fftResultPholder)
    );
  } else if (src) {
    runFftWhenImageReady(dom.imgResult, dom.imgFftResult, dom.fftResultPholder);
  } else {
    dom.imgFftResult.classList.add('hidden');
    dom.fftResultPholder.textContent = 'No transformed image';
    dom.fftResultPholder.classList.remove('hidden');
  }

  // ── FFT — Warped Phase ────────────────────────
  if (data.paths?.fft_phase_proc_path) {
    showFFTImage(
      dom.imgFftPhaseResult,
      dom.fftPhaseResultPholder,
      '/' + data.paths.fft_phase_proc_path + '?t=' + Date.now(),
      () => runFftWhenImageReady(dom.imgResult, dom.imgFftPhaseResult, dom.fftPhaseResultPholder, true)
    );
  } else if (src) {
    runFftWhenImageReady(dom.imgResult, dom.imgFftPhaseResult, dom.fftPhaseResultPholder, true);
  } else {
    if (dom.imgFftPhaseResult) dom.imgFftPhaseResult.classList.add('hidden');
    if (dom.fftPhaseResultPholder) {
      dom.fftPhaseResultPholder.textContent = 'No transformed image';
      dom.fftPhaseResultPholder.classList.remove('hidden');
    }
  }

  // ── Grayscale overlays ────────────────────────
  const grayOrigPath   = data.paths?.grayscale_path        ?? null;
  const grayResultPath = data.paths?.grayscale_result_path ?? null;
  setGrayscaleData(grayOrigPath, grayResultPath);

  // ── Landmark overlay ──────────────────────────
  if (data.landmark_detection?.points) {
    const scale   = data.preprocess?.scale    ?? 1;
    const padLeft = data.preprocess?.pad_left ?? 0;
    const padTop  = data.preprocess?.pad_top  ?? 0;
    const origW   = data.metadata?.original_width  ?? 512;
    const origH   = data.metadata?.original_height ?? 512;

    dom.canvasLmOrig.style.zIndex = '10';
    dom.canvasLmOrig.width  = origW;
    dom.canvasLmOrig.height = origH;

    drawLandmarksLetterbox(
      dom.canvasLmOrig,
      data.landmark_detection.points,
      scale, padLeft, padTop, origW, origH
    );
    dom.canvasLmOrig.classList.toggle('hidden', !state.showLandmarks);
  }

  // ── Metrics ───────────────────
  const mse  = data.metrics?.mse  ?? null;
  const psnr = data.metrics?.psnr ?? null;
  const ssim = data.metrics?.ssim ?? null;

  updateMetrics(mse, psnr, ssim, elapsed);

  // ── Energy Distribution Tablosu ─────────────
  if (data.metrics) {
    const m = data.metrics;
    const fmtE = v => v != null ? Number(v).toExponential(3) : '—';
    const fmtR = v => v != null ? Number(v).toFixed(4) : '—';

    const tbl = document.getElementById('metrics-table')?.querySelector('tbody');
    if (tbl) {
      // Önceki energy satırlarını temizle
      tbl.querySelectorAll('.energy-row').forEach(r => r.remove());

      const rows = [
        ['High/Low Ratio (Orig)', fmtR(m.energy_ratio_orig)],
        ['High/Low Ratio (Proc)', fmtR(m.energy_ratio_proc)],
        ['High Energy (Orig)',    fmtE(m.high_energy_orig)],
        ['High Energy (Proc)',    fmtE(m.high_energy_proc)],
        ['Low Energy (Orig)',     fmtE(m.low_energy_orig)],
        ['Low Energy (Proc)',     fmtE(m.low_energy_proc)],
      ];

      rows.forEach(([label, value]) => {
        const tr = document.createElement('tr');
        tr.className = 'energy-row border-b border-surface-700/50 hover:bg-surface-700/30 transition-colors';
        tr.innerHTML = `
          <td class="py-2 text-slate-400 font-medium text-[11px]">${label}</td>
          <td class="py-2 text-right font-mono text-slate-300 text-[11px]">${value}</td>
          <td class="py-2 text-right"><span class="text-slate-600 text-[10px]">—</span></td>
        `;
        tbl.appendChild(tr);
      });
    }
  }

  setStatus('Done', 'success');
  showToast('Processing complete!', 'success');
  dom.btnExport.disabled = false;
  setProcessingUI(false);
}

/* ═══════════════════════════════════════════════
   Metrics rendering
═══════════════════════════════════════════════ */
function updateMetrics(mse, psnr, ssim, elapsed) {
  const fmtNum = (v, d = 4) => v != null ? Number(v).toFixed(d) : '—';

  dom.metricMse.textContent  = fmtNum(mse,  2);
  dom.metricPsnr.textContent = psnr != null ? `${fmtNum(psnr, 2)} dB` : '—';
  dom.metricSsim.textContent = fmtNum(ssim, 4);
  dom.metricsStrip.style.opacity = '1';

  if (psnr != null) {
    const cls = psnr > 40 ? 'metric-good' : psnr > 30 ? 'metric-warn' : 'metric-bad';
    dom.metricPsnr.className = `text-xl font-bold font-mono ${cls}`;
  }
  if (ssim != null) {
    const cls = ssim > 0.9 ? 'metric-good' : ssim > 0.7 ? 'metric-warn' : 'metric-bad';
    dom.metricSsim.className = `text-xl font-bold font-mono ${cls}`;
  }

  dom.tblMse.textContent  = fmtNum(mse,  4);
  dom.tblPsnr.textContent = psnr != null ? `${fmtNum(psnr, 2)} dB` : '—';
  dom.tblSsim.textContent = fmtNum(ssim, 4);
  dom.tblTime.textContent = elapsed != null ? `${elapsed} ms` : '—';

  dom.tblMseRating.innerHTML  = ratingBadge('mse',  mse);
  dom.tblPsnrRating.innerHTML = ratingBadge('psnr', psnr);
  dom.tblSsimRating.innerHTML = ratingBadge('ssim', ssim);
}

function ratingBadge(metric, value) {
  if (value == null) return '<span class="text-slate-600">N/A</span>';
  let label, cls;
  if (metric === 'mse') {
    label = value < 50 ? 'Excellent' : value < 200 ? 'Good' : 'Poor';
    cls   = value < 50 ? 'metric-good' : value < 200 ? 'metric-warn' : 'metric-bad';
  } else if (metric === 'psnr') {
    label = value > 40 ? 'Excellent' : value > 30 ? 'Good' : 'Poor';
    cls   = value > 40 ? 'metric-good' : value > 30 ? 'metric-warn' : 'metric-bad';
  } else {
    label = value > 0.9 ? 'Excellent' : value > 0.7 ? 'Good' : 'Poor';
    cls   = value > 0.9 ? 'metric-good' : value > 0.7 ? 'metric-warn' : 'metric-bad';
  }
  return `<span class="${cls} font-medium">${label}</span>`;
}

/* ═══════════════════════════════════════════════
   Landmark drawing (canvas overlay)
═══════════════════════════════════════════════ */
function drawLandmarks(canvas, points) {
  if (!points || !points.length) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const W = canvas.width;
  const H = canvas.height;

  points.forEach(point => {
    const px = (point.x / 512) * W;
    const py = (point.y / 512) * H;
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fillStyle   = 'rgba(26,82,255,0.9)';
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth   = 1;
    ctx.fill();
    ctx.stroke();
  });
}

function drawLandmarksLetterbox(canvas, points, scale, padLeft, padTop, origW, origH) {
  if (!points || !points.length) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const W = canvas.width;
  const H = canvas.height;

  points.forEach(point => {
    const px = ((point.x - padLeft) / scale) * (W / origW);
    const py = ((point.y - padTop)  / scale) * (H / origH);
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fillStyle   = 'rgba(26,82,255,0.9)';
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth   = 1;
    ctx.fill();
    ctx.stroke();
  });
}

/* ═══════════════════════════════════════════════
   UI state helpers
═══════════════════════════════════════════════ */
function setProcessingUI(active) {
  dom.btnProcess.disabled = active;
  if (dom.btnAiExpression) dom.btnAiExpression.disabled = active || !state.sessionId || !state.imageId;
  dom.btnProcessLabel.classList.toggle('hidden', active);
  dom.btnProcessLabel.classList.toggle('flex',   !active);
  dom.btnProcessSpin.classList.toggle('hidden',  !active);
  dom.btnProcessSpin.classList.toggle('flex',    active);
  dom.resultSpinner.classList.toggle('hidden',   !active);
  dom.processingBadge.classList.toggle('hidden', !active);
  dom.processingBadge.classList.toggle('flex',   active);
}

/* ═══════════════════════════════════════════════
   Analytics drawer toggle
═══════════════════════════════════════════════ */
let analyticsOpen = false;

dom.btnAnalytics.addEventListener('click', () => {
  analyticsOpen = !analyticsOpen;
  dom.analyticsDrawer.classList.toggle('collapsed', !analyticsOpen);
  dom.analyticsDrawer.classList.toggle('expanded',   analyticsOpen);
  dom.analyticsChevron.style.transform = analyticsOpen ? 'rotate(180deg)' : '';
  dom.btnAnalytics.setAttribute('aria-expanded', String(analyticsOpen));
});

/* ═══════════════════════════════════════════════
   Export Report
═══════════════════════════════════════════════ */
dom.btnExport.addEventListener('click', () => {
  if (!state.resultData) {
    showToast('No result data to export yet.', 'warning');
    return;
  }
  try {
    exportReport();
  } catch (err) {
    console.error('[FaceWarp] Export error:', err);
    showToast('Export failed: ' + err.message, 'error', 5000);
  }
});

function exportReport() {
  if (!window.jspdf) {
    showToast('PDF library not loaded. Please refresh the page.', 'error');
    return;
  }
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const data = state.resultData;
  const ts = new Date().toLocaleString();
  const pw = doc.internal.pageSize.getWidth();
  const ph = doc.internal.pageSize.getHeight();
  const margin = 15;
  const contentW = pw - margin * 2;
  let y = margin;

  /* ── Helper: add a new page if needed ── */
  function checkPage(need) {
    if (y + need > ph - 20) {
      doc.addPage();
      y = margin;
    }
  }

  /* ── Helper: draw section title ── */
  function sectionTitle(title) {
    checkPage(14);
    doc.setFillColor(26, 82, 255);
    doc.rect(margin, y, contentW, 8, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(255, 255, 255);
    doc.text(title, margin + 3, y + 5.5);
    y += 12;
  }

  /* ── Helper: draw key-value row ── */
  function kvRow(key, value, isAlt) {
    checkPage(8);
    if (isAlt) {
      doc.setFillColor(240, 244, 255);
      doc.rect(margin, y - 1, contentW, 7, 'F');
    }
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(80, 80, 80);
    doc.text(key, margin + 3, y + 3);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(30, 30, 30);
    doc.text(String(value), pw - margin - 3, y + 3, { align: 'right' });
    y += 7;
  }

  /* ── Helper: get image data URL from an img element ── */
  function imgToDataUrl(imgEl, maxW, maxH) {
    if (!imgEl || !imgEl.naturalWidth) return null;
    const canvas = document.createElement('canvas');
    let w = imgEl.naturalWidth;
    let h = imgEl.naturalHeight;
    const scale = Math.min(maxW / w, maxH / h, 1);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgEl, 0, 0, w, h);
    return { dataUrl: canvas.toDataURL('image/jpeg', 0.92), w, h };
  }

  /* ═══════════════════════════════════════
     HEADER
  ═══════════════════════════════════════ */
  doc.setFillColor(11, 15, 26);
  doc.rect(0, 0, pw, 35, 'F');
  doc.setFillColor(26, 82, 255);
  doc.rect(0, 33, pw, 2, 'F');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.setTextColor(255, 255, 255);
  doc.text('FaceWarp Lab', margin, 16);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(150, 170, 200);
  doc.text('Facial Image Warping & Aging — Analysis Report', margin, 23);
  doc.text(ts, pw - margin, 23, { align: 'right' });

  doc.setFontSize(8);
  doc.setTextColor(100, 120, 160);
  doc.text(`Session: ${data.session_id || '—'}  |  Image: ${data.image_id || '—'}`, margin, 29);

  y = 42;

  /* ═══════════════════════════════════════
     PARAMETERS
  ═══════════════════════════════════════ */
  sectionTitle('SESSION PARAMETERS');
  const params = [
    ['Operation Mode', state.mode === 'aging' ? 'Aging Simulation' : 'Expression Warping'],
    ['Aging Intensity', fmt(state.agingIntensity)],
    ['Smile Intensity', fmt(state.smileIntensity)],
    ['Eyebrow Height', fmt(state.eyebrowHeight)],
    ['Lip Widening', fmt(state.lipIntensity)],
    ['Face Slimming', fmt(state.faceSlimming)],
    ['Landmark Overlay', state.showLandmarks ? 'Enabled' : 'Disabled'],
    ['Grayscale View', state.showGrayscale ? 'Enabled' : 'Disabled'],
  ];
  params.forEach(([k, v], i) => kvRow(k, v, i % 2 === 0));
  y += 4;

  /* ═══════════════════════════════════════
     QUALITY METRICS
  ═══════════════════════════════════════ */
  sectionTitle('QUALITY METRICS');
  const fmtNum = (v, d = 4) => v != null ? Number(v).toFixed(d) : 'N/A';
  const mse = data.metrics?.mse;
  const psnr = data.metrics?.psnr;
  const ssim = data.metrics?.ssim;

  function ratingText(metric, val) {
    if (val == null) return 'N/A';
    if (metric === 'psnr') return val > 40 ? '[Excellent]' : val > 30 ? '[Good]' : '[Low]';
    if (metric === 'ssim') return val > 0.95 ? '[Excellent]' : val > 0.8 ? '[Good]' : '[Low]';
    if (metric === 'mse') return val < 50 ? '[Low - Good]' : val < 200 ? '[Moderate]' : '[High]';
    return '-';
  }

  const metrics = [
    ['MSE (Mean Squared Error)', fmtNum(mse, 2), ratingText('mse', mse)],
    ['PSNR (Peak Signal-to-Noise)', fmtNum(psnr, 2) + ' dB', ratingText('psnr', psnr)],
    ['SSIM (Structural Similarity)', fmtNum(ssim, 4), ratingText('ssim', ssim)],
    ['Processing Time', (data.elapsed || '—') + ' ms', '—'],
  ];

  // Table header
  checkPage(10);
  doc.setFillColor(30, 42, 58);
  doc.rect(margin, y - 1, contentW, 7, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(255, 255, 255);
  doc.text('Metric', margin + 3, y + 3);
  doc.text('Value', margin + contentW * 0.55, y + 3);
  doc.text('Rating', pw - margin - 3, y + 3, { align: 'right' });
  y += 8;

  metrics.forEach(([label, value, rating], i) => {
    checkPage(8);
    if (i % 2 === 0) {
      doc.setFillColor(245, 247, 255);
      doc.rect(margin, y - 1, contentW, 7, 'F');
    }
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(60, 60, 60);
    doc.text(label, margin + 3, y + 3);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(20, 20, 20);
    doc.text(value, margin + contentW * 0.55, y + 3);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(100, 100, 100);
    doc.text(rating, pw - margin - 3, y + 3, { align: 'right' });
    y += 7;
  });
  y += 4;

  /* ═══════════════════════════════════════
     ENERGY ANALYSIS
  ═══════════════════════════════════════ */
  if (data.metrics) {
    sectionTitle('FREQUENCY-DOMAIN ENERGY ANALYSIS');
    const m = data.metrics;
    const fmtE = v => v != null ? Number(v).toExponential(3) : 'N/A';
    const fmtR = v => v != null ? Number(v).toFixed(4) : 'N/A';
    const energyRows = [
      ['High/Low Energy Ratio (Original)', fmtR(m.energy_ratio_orig)],
      ['High/Low Energy Ratio (Warped)', fmtR(m.energy_ratio_proc)],
      ['Total Energy (Original)', fmtE(m.total_energy_orig)],
      ['Total Energy (Warped)', fmtE(m.total_energy_proc)],
      ['High-Freq Energy (Original)', fmtE(m.high_energy_orig)],
      ['High-Freq Energy (Warped)', fmtE(m.high_energy_proc)],
      ['Low-Freq Energy (Original)', fmtE(m.low_energy_orig)],
      ['Low-Freq Energy (Warped)', fmtE(m.low_energy_proc)],
    ];
    energyRows.forEach(([k, v], i) => kvRow(k, v, i % 2 === 0));
    y += 4;
  }

  /* ═══════════════════════════════════════
     IMAGES — Original & Warped
  ═══════════════════════════════════════ */
  checkPage(90);
  sectionTitle('IMAGE COMPARISON');

  const maxImgW = (contentW - 6) / 2;
  const maxImgH = 70;

  const origInfo = imgToDataUrl(dom.imgOriginal, maxImgW * 3, maxImgH * 3);
  const resInfo  = imgToDataUrl(dom.imgResult,   maxImgW * 3, maxImgH * 3);

  if (origInfo || resInfo) {
    const imgY = y;
    // Labels
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(100, 100, 100);

    if (origInfo) {
      doc.text('ORIGINAL', margin + maxImgW / 2, y, { align: 'center' });
      const dispW = Math.min(origInfo.w / 3, maxImgW);
      const dispH = Math.min(origInfo.h / 3, maxImgH);
      const offsetX = margin + (maxImgW - dispW) / 2;
      doc.addImage(origInfo.dataUrl, 'JPEG', offsetX, y + 2, dispW, dispH);
    }

    if (resInfo) {
      const rightStart = margin + maxImgW + 6;
      doc.text('WARPED', rightStart + maxImgW / 2, y, { align: 'center' });
      const dispW = Math.min(resInfo.w / 3, maxImgW);
      const dispH = Math.min(resInfo.h / 3, maxImgH);
      const offsetX = rightStart + (maxImgW - dispW) / 2;
      doc.addImage(resInfo.dataUrl, 'JPEG', offsetX, y + 2, dispW, dispH);
    }

    y += maxImgH + 8;
  }

  /* ═══════════════════════════════════════
     FFT SPECTRUM IMAGES
  ═══════════════════════════════════════ */
  const fftOrigInfo = imgToDataUrl(dom.imgFftOrig,   maxImgW * 3, maxImgH * 3);
  const fftResInfo  = imgToDataUrl(dom.imgFftResult,  maxImgW * 3, maxImgH * 3);

  if (fftOrigInfo || fftResInfo) {
    checkPage(90);
    sectionTitle('FFT MAGNITUDE SPECTRUM');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(100, 100, 100);

    if (fftOrigInfo) {
      doc.text('ORIGINAL SPECTRUM', margin + maxImgW / 2, y, { align: 'center' });
      const dispW = Math.min(fftOrigInfo.w / 3, maxImgW);
      const dispH = Math.min(fftOrigInfo.h / 3, maxImgH);
      const offsetX = margin + (maxImgW - dispW) / 2;
      doc.addImage(fftOrigInfo.dataUrl, 'JPEG', offsetX, y + 2, dispW, dispH);
    }

    if (fftResInfo) {
      const rightStart = margin + maxImgW + 6;
      doc.text('WARPED SPECTRUM', rightStart + maxImgW / 2, y, { align: 'center' });
      const dispW = Math.min(fftResInfo.w / 3, maxImgW);
      const dispH = Math.min(fftResInfo.h / 3, maxImgH);
      const offsetX = rightStart + (maxImgW - dispW) / 2;
      doc.addImage(fftResInfo.dataUrl, 'JPEG', offsetX, y + 2, dispW, dispH);
    }

    y += maxImgH + 8;
  }

  /* ═══════════════════════════════════════
     FFT PHASE SPECTRUM IMAGES
  ═══════════════════════════════════════ */
  const fftPhaseOrigInfo = imgToDataUrl(dom.imgFftPhaseOrig,   maxImgW * 3, maxImgH * 3);
  const fftPhaseResInfo  = imgToDataUrl(dom.imgFftPhaseResult,  maxImgW * 3, maxImgH * 3);

  if (fftPhaseOrigInfo || fftPhaseResInfo) {
    checkPage(90);
    sectionTitle('FFT PHASE SPECTRUM');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(100, 100, 100);

    if (fftPhaseOrigInfo) {
      doc.text('ORIGINAL PHASE', margin + maxImgW / 2, y, { align: 'center' });
      const dispW = Math.min(fftPhaseOrigInfo.w / 3, maxImgW);
      const dispH = Math.min(fftPhaseOrigInfo.h / 3, maxImgH);
      const offsetX = margin + (maxImgW - dispW) / 2;
      doc.addImage(fftPhaseOrigInfo.dataUrl, 'JPEG', offsetX, y + 2, dispW, dispH);
    }

    if (fftPhaseResInfo) {
      const rightStart = margin + maxImgW + 6;
      doc.text('WARPED PHASE', rightStart + maxImgW / 2, y, { align: 'center' });
      const dispW = Math.min(fftPhaseResInfo.w / 3, maxImgW);
      const dispH = Math.min(fftPhaseResInfo.h / 3, maxImgH);
      const offsetX = rightStart + (maxImgW - dispW) / 2;
      doc.addImage(fftPhaseResInfo.dataUrl, 'JPEG', offsetX, y + 2, dispW, dispH);
    }

    y += maxImgH + 8;
  }

  /* ═══════════════════════════════════════
     PIPELINE STATUS
  ═══════════════════════════════════════ */
  if (data.pipeline) {
    checkPage(40);
    sectionTitle('PIPELINE STATUS');
    const pipelineRows = [
      ['Upload', data.pipeline.upload || '—'],
      ['Decode', data.pipeline.decode || '—'],
      ['Preprocess', data.pipeline.preprocess || '—'],
      ['Face Detection', data.pipeline.face_detection || '—'],
      ['Landmark Detection', data.pipeline.landmark_detection || '—'],
    ];
    pipelineRows.forEach(([k, v], i) => kvRow(k, v, i % 2 === 0));
    y += 4;
  }

  /* ═══════════════════════════════════════
     FOOTER (on every page)
  ═══════════════════════════════════════ */
  const totalPages = doc.internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    // Bottom line
    doc.setDrawColor(26, 82, 255);
    doc.setLineWidth(0.5);
    doc.line(margin, ph - 12, pw - margin, ph - 12);
    // Footer text
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7);
    doc.setTextColor(150, 150, 150);
    doc.text('FaceWarp Lab — Facial Image Warping & Aging System  |  CENG387', margin, ph - 8);
    doc.text(`Page ${i} / ${totalPages}`, pw - margin, ph - 8, { align: 'right' });
  }

  /* ═══════════════════════════════════════
     SAVE
  ═══════════════════════════════════════ */
  const filename = `FaceWarp_Report_${Date.now()}.pdf`;
  doc.save(filename);
  showToast('PDF report downloaded!', 'success');
}

/* ═══════════════════════════════════════════════
   Gallery System
═══════════════════════════════════════════════ */
const galleryBtn   = $('btn-gallery');
const galleryModal = $('gallery-modal');
const galleryClose = $('btn-close-gallery');
const galleryGrid  = $('gallery-grid');
const galleryEmpty = $('gallery-empty');

// Modal açma/kapama fonksiyonları
function openGallery() {
  galleryModal.classList.remove('hidden');
  setTimeout(() => {
    galleryModal.classList.remove('opacity-0');
    const inner = galleryModal.querySelector('.glass');
    if (inner) inner.classList.remove('scale-95');
  }, 10);
  loadGallery();
}

function closeGallery() {
  galleryModal.classList.add('opacity-0');
  const inner = galleryModal.querySelector('.glass');
  if (inner) inner.classList.add('scale-95');
  setTimeout(() => {
    galleryModal.classList.add('hidden');
  }, 300);
}

if(galleryBtn) galleryBtn.addEventListener('click', openGallery);
if(galleryClose) galleryClose.addEventListener('click', closeGallery);

// Modal dışına tıklayınca kapanması için
galleryModal.addEventListener('click', (e) => {
    if(e.target === galleryModal) closeGallery();
});

async function loadGallery() {
  galleryGrid.innerHTML = '<div class="col-span-full flex justify-center py-10"><div class="spinner border-brand-500"></div></div>';
  galleryEmpty.classList.add('hidden');

  try {
    const res = await fetch('/api/gallery');
    const data = await res.json();

    if (!data.success || !data.items || data.items.length === 0) {
      galleryGrid.innerHTML = '';
      galleryEmpty.classList.remove('hidden');
      return;
    }

    galleryGrid.innerHTML = '';
    data.items.forEach(item => {
      const dateObj = new Date(item.created_at * 1000);
      const dateStr = dateObj.toLocaleDateString('en-GB') + ' ' + dateObj.toLocaleTimeString('en-GB', {hour: '2-digit', minute:'2-digit'});
      const shortId = item.image_id.split('_')[1]?.substring(0, 8) || 'img';
      const hasOriginal = item.original_path && item.original_path !== "";

      const card = document.createElement('div');
      card.className = 'group relative glass rounded-xl overflow-hidden border border-surface-600/50 hover:border-brand-500 hover:shadow-[0_0_20px_rgba(26,82,255,0.2)] transition-all duration-300 cursor-pointer';

      card.innerHTML = `
        <div class="relative w-full aspect-square bg-surface-800">
          <img src="/${item.preprocessed_path}" class="absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${hasOriginal ? 'group-hover:opacity-0' : ''}" loading="lazy" />
          ${hasOriginal ? `
          <img src="/${item.original_path}" class="absolute inset-0 w-full h-full object-cover opacity-0 transition-opacity duration-300 group-hover:opacity-100" loading="lazy" onerror="this.style.display='none';" />
          <div class="absolute top-2 right-2 bg-surface-900/90 backdrop-blur px-2 py-1 rounded-md text-[9px] font-medium text-slate-300 border border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            Holding: Original
          </div>
          ` : ''}
           <!-- Silme butonu — sağ üst köşe -->
        <button class="btn-delete-card absolute top-2 left-2 w-7 h-7 rounded-lg bg-red-600/80 hover:bg-red-500 border border-red-500/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity z-10"
          data-session="${item.session_id}" data-image="${item.image_id}" title="Delete">
          <svg class="w-3.5 h-3.5 text-white pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>

      <div class="p-3 bg-surface-900/50 flex justify-between items-center">
        <div>
          <p class="text-[11px] font-bold text-slate-300 font-mono truncate" title="${item.image_id}">#${shortId}</p>
          <div class="flex items-center gap-1.5 mt-1.5">
            <svg class="w-3 h-3 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            <p class="text-[10px] text-slate-400">${dateStr}</p>
          </div>
        </div>
        <div class="w-7 h-7 rounded-full bg-brand-600/20 border border-brand-500/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity transform scale-75 group-hover:scale-100">
          <svg class="w-3.5 h-3.5 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" /></svg>
        </div>
      </div>
      `;

      card.querySelector('.btn-delete-card').addEventListener('click', async (e) => {
        e.stopPropagation(); // kartın lightbox açmasını engelle
        const sid = e.currentTarget.dataset.session;
        const iid = e.currentTarget.dataset.image;

        if (!confirm('Bu resmi silmek istediğinizden emin misiniz?')) return;

        try {
          const res = await fetch(`/api/gallery/${sid}/${iid}`, { method: 'DELETE' });
          const data = await res.json();
          if (data.success) {
            card.remove(); // kartı DOM'dan kaldır
            showToast('Image deleted.', 'success');
            if (galleryGrid.children.length === 0) {
              galleryEmpty.classList.remove('hidden');
            }
          } else {
            showToast('Delete failed.', 'error');
          }
        } catch {
          showToast('Delete failed.', 'error');
        }
      });
      card.addEventListener('click', () => {
        openLightbox('/' + item.preprocessed_path, hasOriginal ? '/' + item.original_path : null);
      });

      galleryGrid.appendChild(card);
    });

  } catch (err) {
    console.error('[FaceWarp] Gallery load error:', err);
    galleryGrid.innerHTML = '<div class="col-span-full text-center text-red-400 py-10 text-sm">Failed to load gallery. Please try again.</div>';
  }
}

/* ═══════════════════════════════════════════════
   Initialise
═══════════════════════════════════════════════ */
function init() {
  dom.valAging.textContent   = fmt(state.agingIntensity);
  dom.valSmile.textContent   = fmt(state.smileIntensity);
  dom.valEyebrow.textContent = fmt(state.eyebrowHeight);
  dom.valLip.textContent     = fmt(state.lipIntensity);
  dom.valSlim.textContent    = fmt(state.faceSlimming);
  bindEditorControls();
  bindTryonControls();
  syncTryonCategoryOptions();
  loadEditorAssets();
  refreshTryonStatus();

  const parameterSection = dom.groupExpression?.parentElement;
  if (
    parameterSection &&
    dom.aiExpressionPanel &&
    dom.aiExpressionPanel.parentElement !== parameterSection
  ) {
    parameterSection.insertBefore(dom.aiExpressionPanel, dom.groupAging);
  }

  // Show only the expression group on startup
  dom.groupExpression.style.display  = '';
  dom.groupAging.style.display       = 'none';
  dom.groupAccessory.style.display   = 'none';
  if (dom.groupTryon) dom.groupTryon.style.display = 'none';
  if (dom.aiExpressionPanel) dom.aiExpressionPanel.style.display = 'none';

  updateParamSnapshot();
  setStatus('Ready', 'idle');

  console.info('[FaceWarp Lab] Initialised — waiting for image upload.');
}

/* ═══════════════════════════════════════════════
   Lightbox System (Full Screen View)
═══════════════════════════════════════════════ */
const lightboxModal       = $('lightbox-modal');
const lightboxImg         = $('lightbox-img');
const btnCloseLightbox    = $('btn-close-lightbox');
const btnLightboxOriginal = $('btn-lightbox-original');

let currentPreprocessed = '';
let currentOriginal     = '';
let showingOriginal     = false;

function openLightbox(preprocessedSrc, originalSrc) {
    currentPreprocessed = preprocessedSrc;
    currentOriginal     = originalSrc;
    showingOriginal     = false;
    
    lightboxImg.src = preprocessedSrc;
    
    // Eğer orijinal fotoğraf varsa butonu göster
    if (originalSrc) {
        btnLightboxOriginal.classList.remove('hidden');
        btnLightboxOriginal.textContent = 'Show Original';
        btnLightboxOriginal.className = 'absolute bottom-6 px-5 py-2.5 bg-surface-900/80 backdrop-blur border border-white/10 rounded-xl text-xs font-semibold text-white hover:bg-brand-600 transition-colors shadow-lg';
    } else {
        btnLightboxOriginal.classList.add('hidden');
    }

    lightboxModal.classList.remove('hidden');
    setTimeout(() => {
        lightboxModal.classList.remove('opacity-0');
        lightboxImg.classList.remove('scale-95');
    }, 10);
}

function closeLightbox() {
    lightboxModal.classList.add('opacity-0');
    lightboxImg.classList.add('scale-95');
    setTimeout(() => {
        lightboxModal.classList.add('hidden');
        lightboxImg.src = '';
    }, 300);
}

if(btnCloseLightbox) btnCloseLightbox.addEventListener('click', closeLightbox);

if(lightboxModal) {
    lightboxModal.addEventListener('click', (e) => {
        if(e.target === lightboxModal || e.target.parentElement === lightboxModal) closeLightbox();
    });
}

if(btnLightboxOriginal) {
    btnLightboxOriginal.addEventListener('click', (e) => {
        e.stopPropagation(); // Tıklamanın arka plana geçip modalı kapatmasını engeller
        showingOriginal = !showingOriginal;
        
        // Yumuşak bir geçiş (fade) efekti
        lightboxImg.style.opacity = '0';
        setTimeout(() => {
            lightboxImg.src = showingOriginal ? currentOriginal : currentPreprocessed;
            btnLightboxOriginal.textContent = showingOriginal ? 'Show Processed' : 'Show Original';
            
            // Buton rengini aktifliğe göre değiştir
            if(showingOriginal) {
                btnLightboxOriginal.classList.add('bg-brand-600', 'border-brand-500');
                btnLightboxOriginal.classList.remove('bg-surface-900/80', 'border-white/10');
            } else {
                btnLightboxOriginal.classList.remove('bg-brand-600', 'border-brand-500');
                btnLightboxOriginal.classList.add('bg-surface-900/80', 'border-white/10');
            }
            
            lightboxImg.style.opacity = '1';
        }, 150);
    });
}

init();
