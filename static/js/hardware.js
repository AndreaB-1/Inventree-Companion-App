// Hardware definitions used by the label maker (ported from v1).

export const METRIC_DIMS = {
  'M1': { pitch: 0.25, af: 2.5, nut_h: 0.8, w_id: 1.1, w_od: 3.0, w_t: 0.3 },
  'M1.2': { pitch: 0.25, af: 3.0, nut_h: 1.0, w_id: 1.3, w_od: 3.5, w_t: 0.3 },
  'M1.4': { pitch: 0.30, af: 3.0, nut_h: 1.2, w_id: 1.5, w_od: 3.5, w_t: 0.3 },
  'M1.6': { pitch: 0.35, af: 3.2, nut_h: 1.3, w_id: 1.7, w_od: 4.0, w_t: 0.3 },
  'M2': { pitch: 0.40, af: 4.0, nut_h: 1.6, w_id: 2.2, w_od: 5.0, w_t: 0.3 },
  'M2.5': { pitch: 0.45, af: 5.0, nut_h: 2.0, w_id: 2.7, w_od: 6.0, w_t: 0.5 },
  'M3': { pitch: 0.50, af: 5.5, nut_h: 2.4, w_id: 3.2, w_od: 7.0, w_t: 0.5 },
  'M3.5': { pitch: 0.60, af: 6.0, nut_h: 2.8, w_id: 3.7, w_od: 8.0, w_t: 0.5 },
  'M4': { pitch: 0.70, af: 7.0, nut_h: 3.2, w_id: 4.3, w_od: 9.0, w_t: 0.8 },
  'M5': { pitch: 0.80, af: 8.0, nut_h: 4.7, w_id: 5.3, w_od: 10.0, w_t: 1.0 },
  'M6': { pitch: 1.00, af: 10.0, nut_h: 5.2, w_id: 6.4, w_od: 12.0, w_t: 1.6 },
  'M7': { pitch: 1.00, af: 11.0, nut_h: 5.5, w_id: 7.4, w_od: 14.0, w_t: 1.6 },
  'M8': { pitch: 1.25, af: 13.0, nut_h: 6.8, w_id: 8.4, w_od: 16.0, w_t: 1.6 },
  'M10': { pitch: 1.50, af: 17.0, nut_h: 8.4, w_id: 10.5, w_od: 20.0, w_t: 2.0 },
  'M12': { pitch: 1.75, af: 19.0, nut_h: 10.8, w_id: 13.0, w_od: 24.0, w_t: 2.5 },
  'M14': { pitch: 2.00, af: 22.0, nut_h: 12.8, w_id: 15.0, w_od: 28.0, w_t: 2.5 },
  'M16': { pitch: 2.00, af: 24.0, nut_h: 14.8, w_id: 17.0, w_od: 30.0, w_t: 3.0 },
  'M18': { pitch: 2.50, af: 27.0, nut_h: 15.0, w_id: 19.0, w_od: 34.0, w_t: 3.0 },
  'M20': { pitch: 2.50, af: 30.0, nut_h: 18.0, w_id: 21.0, w_od: 37.0, w_t: 3.0 },
  'M22': { pitch: 2.50, af: 32.0, nut_h: 17.5, w_id: 23.0, w_od: 39.0, w_t: 3.0 },
  'M24': { pitch: 3.00, af: 36.0, nut_h: 21.5, w_id: 25.0, w_od: 44.0, w_t: 4.0 },
  'M27': { pitch: 3.00, af: 41.0, nut_h: 22.5, w_id: 28.0, w_od: 50.0, w_t: 4.0 },
  'M30': { pitch: 3.50, af: 46.0, nut_h: 24.0, w_id: 31.0, w_od: 56.0, w_t: 4.0 },
  'M36': { pitch: 4.00, af: 55.0, nut_h: 29.4, w_id: 37.0, w_od: 66.0, w_t: 5.0 },
  'M42': { pitch: 4.50, af: 65.0, nut_h: 34.0, w_id: 43.0, w_od: 78.0, w_t: 6.0 },
  'M48': { pitch: 5.00, af: 75.0, nut_h: 38.0, w_id: 50.0, w_od: 92.0, w_t: 6.0 },
};

export const THREAD_SIZES = Object.keys(METRIC_DIMS);
export const MATERIALS = ['A2-304 Stainless', 'A4-316 Stainless', 'Carbon Steel', 'Alloy Steel', 'High Tensile Steel (Grade 8.8)', 'High Tensile Steel (Grade 10.9)', 'High Tensile Steel (Grade 12.9)', 'Zinc Plated Steel', 'Hot-Dip Galvanised Steel', 'Black Oxide Steel', 'Brass', 'Aluminium', 'Titanium Grade 2', 'Nylon (PA66)', 'PTFE'];

// Field types: select (opts) | number. `dyn` marks lists replaced by server config.
export const HW_TYPES = {
  screw: { label: 'Screw / Bolt', fields: [
    { id: 'thread_size', label: 'Thread size', type: 'select', dyn: 'sizes', opts: THREAD_SIZES, def: 'M4' },
    { id: 'length_mm', label: 'Length (mm)', type: 'number', def: '16', min: 1, max: 500, step: 1 },
    { id: 'pitch_mm', label: 'Pitch (mm)', type: 'number', def: '0.70', step: 0.05, hint: 'Auto-filled from size' },
    { id: 'head_type', label: 'Head type', type: 'select', opts: ['Socket Cap', 'Button', 'Pan', 'Countersunk', 'Hex', 'Flange Hex', 'Truss', 'Set Screw'], def: 'Socket Cap' },
    { id: 'drive', label: 'Drive', type: 'select', opts: ['Hex (Allen)', 'Torx', 'Phillips', 'Slotted', 'Pozidriv', 'Square', 'None'], def: 'Hex (Allen)' },
    { id: 'material', label: 'Material', type: 'select', dyn: 'materials', opts: MATERIALS, def: 'A2-304 Stainless' },
    { id: 'grade', label: 'Grade / class', type: 'select', opts: ['A2-70', 'A4-80', 'A2-50', '8.8', '10.9', '12.9', '4.8', '5.8'], def: 'A2-70' },
    { id: 'standard', label: 'Standard', type: 'select', dyn: 'standards', opts: ['ISO 4762', 'DIN 912', 'ISO 7380', 'DIN 7380', 'ISO 10642', 'DIN 933', 'ISO 4014', 'ISO 1207'], def: 'ISO 4762', hint: 'Sets head & drive for known standards' },
    { id: 'finish', label: 'Finish', type: 'select', opts: ['Plain', 'Zinc Plated', 'Black Oxide', 'Nickel Plated', 'Hot Dip Galv', 'Anodized'], def: 'Plain' },
    { id: 'thread_coverage', label: 'Thread coverage', type: 'select', opts: ['Full Thread', 'Partial Thread'], def: 'Full Thread' },
  ] },
  nut: { label: 'Nut', fields: [
    { id: 'thread_size', label: 'Thread size', type: 'select', dyn: 'sizes', opts: THREAD_SIZES, def: 'M4' },
    { id: 'nut_type', label: 'Nut type', type: 'select', opts: ['Hex', 'Hex Thin/Jam', 'Nyloc', 'Flanged', 'Flanged Nyloc', 'Castle', 'Wing', 'Cap/Dome', 'T-nut'], def: 'Hex' },
    { id: 'height_mm', label: 'Height (mm)', type: 'number', def: '3.2', step: 0.1, hint: 'Auto-filled' },
    { id: 'af_mm', label: 'Across flats (mm)', type: 'number', def: '7.0', step: 0.1, hint: 'Auto-filled' },
    { id: 'material', label: 'Material', type: 'select', dyn: 'materials', opts: MATERIALS, def: 'A2-304 Stainless' },
    { id: 'grade', label: 'Grade', type: 'select', opts: ['A2-70', 'A4-80', 'A2-50', '5', '6', '8', '10'], def: 'A2-70' },
    { id: 'standard', label: 'Standard', type: 'select', dyn: 'standards', opts: ['ISO 4032', 'DIN 934', 'ISO 4035', 'ISO 7042', 'ISO 10511'], def: 'ISO 4032' },
  ] },
  washer: { label: 'Washer', fields: [
    { id: 'fits_size', label: 'Fits bolt size', type: 'select', dyn: 'sizes', opts: THREAD_SIZES, def: 'M4' },
    { id: 'washer_type', label: 'Washer type', type: 'select', opts: ['Plain (Form A)', 'Plain (Form B/Large)', 'Split Lock', 'Star Lock (Int)', 'Star Lock (Ext)', 'Fender', 'Spring', 'Belleville', 'Nord-Lock'], def: 'Plain (Form A)' },
    { id: 'id_mm', label: 'Inner dia (mm)', type: 'number', def: '4.3', step: 0.1, hint: 'Auto-filled' },
    { id: 'od_mm', label: 'Outer dia (mm)', type: 'number', def: '9.0', step: 0.1, hint: 'Auto-filled' },
    { id: 'thickness_mm', label: 'Thickness (mm)', type: 'number', def: '0.8', step: 0.05, hint: 'Auto-filled' },
    { id: 'material', label: 'Material', type: 'select', dyn: 'materials', opts: MATERIALS.slice(0, -1), def: 'A2-304 Stainless' },
    { id: 'standard', label: 'Standard', type: 'select', dyn: 'standards', opts: ['ISO 7089', 'ISO 7090', 'DIN 125', 'ISO 7093', 'DIN 9021'], def: 'ISO 7089' },
  ] },
  standoff: { label: 'Standoff', fields: [
    { id: 'thread_size', label: 'Thread size', type: 'select', dyn: 'sizes', opts: ['M2', 'M2.5', 'M3', 'M4', 'M5', 'M6', 'M8'], def: 'M3' },
    { id: 'length_mm', label: 'Length (mm)', type: 'number', def: '10', min: 1, max: 200, step: 1 },
    { id: 'standoff_type', label: 'Type', type: 'select', opts: ['Hex F-F', 'Hex M-F', 'Hex M-M', 'Round F-F', 'PCB'], def: 'Hex F-F' },
    { id: 'body_shape', label: 'Body shape', type: 'select', opts: ['Hex', 'Round'], def: 'Hex' },
    { id: 'material', label: 'Material', type: 'select', dyn: 'materials', opts: ['Brass', 'Aluminum', 'A2-304 Stainless', 'Carbon Steel', 'Nylon'], def: 'Brass' },
  ] },
  rivet: { label: 'Rivet', fields: [
    { id: 'rivet_type', label: 'Type', type: 'select', opts: ['Blind/Pop', 'Solid', 'Drive', 'Tubular'], def: 'Blind/Pop' },
    { id: 'diameter_mm', label: 'Diameter (mm)', type: 'select', opts: ['2.4', '3.2', '4.0', '4.8', '5.0', '6.4'], def: '3.2' },
    { id: 'grip_min', label: 'Grip min (mm)', type: 'number', def: '', step: 0.5 },
    { id: 'grip_max', label: 'Grip max (mm)', type: 'number', def: '', step: 0.5 },
    { id: 'head_type', label: 'Head type', type: 'select', opts: ['Dome', 'Countersunk', 'Large Flange', 'Truss'], def: 'Dome' },
    { id: 'body_material', label: 'Material', type: 'select', opts: ['Aluminum', 'Carbon Steel', 'A2-304 Stainless', 'Copper', 'Monel'], def: 'Aluminum' },
  ] },
  pin: { label: 'Dowel pin', fields: [
    { id: 'diameter_mm', label: 'Diameter (mm)', type: 'select', opts: ['2', '3', '4', '5', '6', '8', '10', '12'], def: '4' },
    { id: 'length_mm', label: 'Length (mm)', type: 'number', def: '20', min: 1, max: 300, step: 1 },
    { id: 'pin_type', label: 'Pin type', type: 'select', opts: ['Solid Cylindrical', 'Roll Pin (Spring)', 'Slotted'], def: 'Solid Cylindrical' },
    { id: 'tolerance', label: 'Tolerance', type: 'select', opts: ['h6 (drive fit)', 'm6 (press fit)', 'g6 (slip fit)', 'f7 (running fit)'], def: 'h6 (drive fit)' },
    { id: 'material', label: 'Material', type: 'select', dyn: 'materials', opts: ['Carbon Steel', 'A2-304 Stainless', 'Hardened Steel', 'Brass'], def: 'Carbon Steel' },
  ] },
  circlip: { label: 'Circlip', fields: [
    { id: 'shaft_diameter_mm', label: 'Shaft / bore dia (mm)', type: 'number', def: '10', min: 1, max: 300, step: 0.5 },
    { id: 'circlip_type', label: 'Type', type: 'select', opts: ['External', 'Internal'], def: 'External' },
    { id: 'material', label: 'Material', type: 'select', dyn: 'materials', opts: ['Carbon Steel', 'A2-304 Stainless', 'Phosphor Bronze'], def: 'Carbon Steel' },
  ] },
  insert: { label: 'Insert', fields: [
    { id: 'thread_size', label: 'Thread size', type: 'select', dyn: 'sizes', opts: ['M2', 'M2.5', 'M3', 'M4', 'M5', 'M6', 'M8', 'M10', 'M12'], def: 'M4' },
    { id: 'length_mm', label: 'Length (mm)', type: 'number', def: '6', min: 1, max: 50, step: 1 },
    { id: 'install_method', label: 'Installation', type: 'select', opts: ['Heat-Set (Brass)', 'Threaded', 'Self-Tapping'], def: 'Heat-Set (Brass)' },
    { id: 'material', label: 'Material', type: 'select', dyn: 'materials', opts: ['Brass', 'A2-304 Stainless', 'Carbon Steel'], def: 'Brass' },
  ] },
  self_tap: { label: 'Self-tapping', fields: [
    { id: 'application', label: 'Application', type: 'select', opts: ['Wood', 'Plastic / Nylon', 'Sheet Metal', 'General Purpose'], def: 'Wood' },
    { id: 'diameter_mm', label: 'Diameter (mm)', type: 'select', opts: ['2', '2.5', '3', '3.5', '4', '4.5', '5', '5.5', '6', '7', '8', '10'], def: '3' },
    { id: 'length_mm', label: 'Length (mm)', type: 'number', def: '16', min: 6, max: 250, step: 1 },
    { id: 'head_type', label: 'Head type', type: 'select', opts: ['Countersunk', 'Pan', 'Truss', 'Wafer', 'Bugle', 'Hex', 'Hex Washer'], def: 'Countersunk' },
    { id: 'head_diameter_mm', label: 'Head diameter (mm)', type: 'number', def: '6', min: 2, max: 30, step: 0.5 },
    { id: 'drive', label: 'Drive', type: 'select', opts: ['Torx T10', 'Torx T15', 'Torx T20', 'Torx T25', 'Torx T30', 'Phillips', 'Square', 'Slotted', 'Nut Driver / Socket'], def: 'Torx T20' },
    { id: 'point_type', label: 'Point type', type: 'select', opts: ['Sharp', 'Type 17 (Auger)', 'Self-Drill (TEK)', 'Thread Forming', 'Blunt (Type B)'], def: 'Sharp' },
    { id: 'finish', label: 'Finish', type: 'select', opts: ['Zinc Plated', 'Cr3+ Zinc', 'Hot Dip Galv', 'Stainless', 'Black Oxide', 'Polymer Coated'], def: 'Zinc Plated' },
    { id: 'thread_coverage', label: 'Thread coverage', type: 'select', opts: ['Full Thread', 'Partial Thread'], def: 'Full Thread' },
  ] },
};
export const HW_TYPE_KEYS = Object.keys(HW_TYPES);

// Known standards → head/drive auto-fill
export const STANDARD_FILL = {
  'DIN 912': { head: 'Socket Cap', drive: 'Hex (Allen)' }, 'ISO 4762': { head: 'Socket Cap', drive: 'Hex (Allen)' },
  'DIN 7984': { head: 'Socket Cap', drive: 'Hex (Allen)' }, 'ISO 7380-1': { head: 'Button', drive: 'Hex (Allen)' },
  'ISO 7380-2': { head: 'Button', drive: 'Hex (Allen)' }, 'DIN 7985': { head: 'Pan', drive: 'Phillips' },
  'ISO 7045': { head: 'Pan', drive: 'Phillips' }, 'DIN 965': { head: 'Countersunk', drive: 'Phillips' },
  'ISO 10642': { head: 'Countersunk', drive: 'Hex (Allen)' }, 'DIN 7991': { head: 'Countersunk', drive: 'Hex (Allen)' },
  'DIN 933': { head: 'Hex', drive: 'None' }, 'DIN 931': { head: 'Hex', drive: 'None' }, 'ISO 4017': { head: 'Hex', drive: 'None' },
  'ASME B18.3': { head: 'Socket Cap', drive: 'Hex (Allen)' }, 'ASME B18.2.1': { head: 'Hex', drive: 'None' },
  'ASME B18.6.3': { head: 'Pan', drive: 'Phillips' }, 'ISO 1207': { head: 'Pan', drive: 'Slotted' }, 'DIN 84': { head: 'Pan', drive: 'Slotted' },
  'DIN 7981': { head: 'Pan', drive: 'Phillips' }, 'DIN 7982': { head: 'Countersunk', drive: 'Phillips' },
  'ISO 14585': { head: 'Pan', drive: 'Torx' }, 'ISO 14586': { head: 'Countersunk', drive: 'Torx' },
};

// Values auto-derived from the metric size
export function sizeAutoFill(type, size) {
  const d = METRIC_DIMS[size]; if (!d) return {};
  if (type === 'screw') return { pitch_mm: d.pitch };
  if (type === 'nut') return { height_mm: d.nut_h, af_mm: d.af };
  if (type === 'washer') return { id_mm: d.w_id, od_mm: d.w_od, thickness_mm: d.w_t };
  return {};
}

const mat = (s) => s.material || s.body_material || '';

export function suggestName(t, s) {
  if (t === 'screw') {
    const sz = (s.thread_size || '') + (s.length_mm ? '×' + s.length_mm : '');
    return [sz, s.head_type, s.thread_coverage === 'Partial Thread' ? 'Partial Thread' : '', 'Screw', s.material].filter(Boolean).join(' ');
  }
  if (t === 'nut') return [s.thread_size, s.nut_type, 'Nut', s.material].filter(Boolean).join(' ');
  if (t === 'washer') return [s.fits_size, s.washer_type, 'Washer', s.material].filter(Boolean).join(' ');
  if (t === 'standoff') return [(s.thread_size || '') + (s.length_mm ? ' ' + s.length_mm + 'mm' : ''), s.standoff_type, 'Standoff', mat(s)].filter(Boolean).join(' ');
  if (t === 'rivet') return [s.diameter_mm ? 'Ø' + s.diameter_mm + 'mm' : '', s.rivet_type, 'Rivet', mat(s)].filter(Boolean).join(' ');
  if (t === 'pin') {
    const sz = s.diameter_mm && s.length_mm ? 'Ø' + s.diameter_mm + '×' + s.length_mm + 'mm' : (s.diameter_mm ? 'Ø' + s.diameter_mm + 'mm' : '');
    return [sz, s.pin_type, s.material].filter(Boolean).join(' ');
  }
  if (t === 'circlip') return [s.shaft_diameter_mm ? 'Ø' + s.shaft_diameter_mm + 'mm' : '', s.circlip_type, 'Circlip', s.material].filter(Boolean).join(' ');
  if (t === 'insert') return [(s.thread_size || '') + (s.length_mm ? ' ' + s.length_mm + 'mm' : ''), s.install_method, 'Insert', s.material].filter(Boolean).join(' ');
  if (t === 'self_tap') {
    const sz = (s.diameter_mm || '') + (s.length_mm ? '×' + s.length_mm + 'mm' : '');
    const app = s.application && s.application !== 'General Purpose' ? s.application : '';
    return [sz, s.head_type, app, s.thread_coverage === 'Partial Thread' ? 'Partial Thread' : '', 'Self-Tapping Screw', s.finish].filter(Boolean).join(' ');
  }
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export function suggestDesc(t, s) {
  const J = (a) => a.filter(Boolean).join(' – ');
  if (t === 'screw') {
    const sz = (s.thread_size || '') + (s.length_mm ? '×' + s.length_mm : '') + (s.pitch_mm ? ' p' + s.pitch_mm : '');
    return J([sz, s.head_type, s.drive, s.thread_coverage === 'Partial Thread' ? 'Partial Thread' : '', 'Screw', s.standard, (s.material || '') + (s.grade ? ' ' + s.grade : '')]);
  }
  if (t === 'nut') return J([(s.thread_size || '') + (s.pitch_mm ? ' p' + s.pitch_mm : ''), s.nut_type, 'Nut', s.standard, (s.material || '') + (s.grade ? ' ' + s.grade : '')]);
  if (t === 'washer') return J([s.fits_size, s.washer_type, 'Washer', s.standard, s.material]);
  if (t === 'standoff') return J([(s.thread_size || '') + (s.length_mm ? ' ' + s.length_mm + 'mm' : ''), s.standoff_type, 'Standoff', mat(s)]);
  if (t === 'rivet') {
    const grip = s.grip_min || s.grip_max ? 'grip ' + (s.grip_min || '') + '–' + (s.grip_max || '') + 'mm' : '';
    return J([s.diameter_mm ? 'Ø' + s.diameter_mm + 'mm' : '', s.rivet_type, 'Rivet', grip, mat(s)]);
  }
  if (t === 'pin') {
    const sz = s.diameter_mm && s.length_mm ? 'Ø' + s.diameter_mm + '×' + s.length_mm + 'mm' : (s.diameter_mm ? 'Ø' + s.diameter_mm + 'mm' : '');
    return J([sz, s.pin_type, s.tolerance, s.material]);
  }
  if (t === 'circlip') return J([s.shaft_diameter_mm ? 'Ø' + s.shaft_diameter_mm + 'mm' : '', s.circlip_type, 'Circlip', s.material]);
  if (t === 'insert') return J([(s.thread_size || '') + (s.length_mm ? ' ' + s.length_mm + 'mm' : ''), s.install_method, 'Thread Insert', s.material]);
  if (t === 'self_tap') {
    const sz = (s.diameter_mm || '') + (s.length_mm ? '×' + s.length_mm + 'mm' : '');
    const hd = s.head_diameter_mm ? 'Ø' + s.head_diameter_mm + 'mm head' : '';
    const app = s.application && s.application !== 'General Purpose' ? s.application + ' ' : '';
    return J([sz, s.head_type, s.drive, s.point_type, s.thread_coverage === 'Partial Thread' ? 'Partial Thread' : '', app + 'Self-Tapping Screw', hd, s.finish]);
  }
  return '';
}

// Lists from the server-side hardware config (with fallbacks)
export function configLists(cfg, type) {
  const ts = (cfg && cfg.thread_sizes) || {};
  const sizes = [...(ts.metric || []), ...(ts.imperial || []), ...(ts.custom || [])];
  const materials = (cfg && cfg.materials) || [];
  const standards = ((cfg && cfg.standards) || []).filter((s) => s.type === type || !s.type).map((s) => s.name);
  return { sizes, materials, standards };
}
