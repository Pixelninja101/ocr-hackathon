/**
 * QR Payload Identification & Parsing Utility
 * ============================================
 * Detects QR content type and extracts structured, human-readable fields
 * where possible - without inventing data or bypassing encryption.
 */

const AADHAAR_OLD_QR_TAG = 'PrintLetterBarcodeData';

function parseAadhaarOldQr(xmlStr) {
  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlStr, 'text/xml');
    const el = doc.querySelector(AADHAAR_OLD_QR_TAG) || doc.documentElement;
    if (!el || el.nodeName === 'parsererror') return null;
    if (!el.hasAttribute('uid') && !el.hasAttribute('name')) return null;
    const get = (attr) => el.getAttribute(attr) || null;
    const uid = get('uid');
    const maskedUid = uid ? 'XXXX XXXX ' + uid.slice(-4) : null;
    const gRaw = get('gender') || get('g');
    const gender = gRaw === 'M' ? 'Male' : gRaw === 'F' ? 'Female' : gRaw === 'T' ? 'Transgender' : gRaw;
    const addrParts = [get('house'), get('street') || get('str'), get('lm'), get('loc'), get('vtc'), get('subdist'), get('dist'), get('state')].filter(Boolean);
    const pincode = get('pc');
    if (pincode) addrParts.push('PIN: ' + pincode);
    return {
      type: 'aadhaar_old',
      label: 'Aadhaar QR (Legacy / Old Format)',
      icon: 'shield',
      fields: [
        maskedUid && { key: 'Aadhaar Number', value: maskedUid, sensitive: true },
        get('name') && { key: 'Name', value: get('name') },
        gender && { key: 'Gender', value: gender },
        get('yob') && { key: 'Year of Birth', value: get('yob') },
        get('dob') && { key: 'Date of Birth', value: get('dob') },
        get('co') && { key: 'Care Of', value: get('co') },
        addrParts.length > 0 && { key: 'Address', value: addrParts.join(', ') },
        get('email') && { key: 'Email Hash', value: get('email'), sensitive: true },
        get('mobile') && { key: 'Mobile Hash', value: get('mobile'), sensitive: true },
      ].filter(Boolean),
      notice: null,
    };
  } catch { return null; }
}

function detectAadhaarSecureQr(str) {
  const trimmed = str.trim();
  if (/^\d+$/.test(trimmed) && trimmed.length > 200) {
    return {
      type: 'aadhaar_secure',
      label: 'Aadhaar Secure QR (Encrypted)',
      icon: 'lock',
      fields: [
        { key: 'Payload Length', value: trimmed.length + ' digits' },
        { key: 'Format', value: 'UIDAI Secure QR — Binary-Encoded / Digitally Signed' },
      ],
      notice: "This is a UIDAI Aadhaar Secure QR Code. The payload is binary-encoded and digitally signed with UIDAI's private key. Demographic data (name, address, photo) is encrypted and cannot be extracted without UIDAI's official verification infrastructure. No decryption is performed.",
    };
  }
  return null;
}

function parseUrl(str) {
  const trimmed = str.trim();
  if (!/^https?:\/\//i.test(trimmed) && !/^ftp:\/\//i.test(trimmed)) return null;
  try {
    const url = new URL(trimmed);
    const fields = [
      { key: 'Full URL', value: trimmed },
      { key: 'Protocol', value: url.protocol.replace(':', '') },
      { key: 'Host', value: url.hostname },
    ];
    if (url.port) fields.push({ key: 'Port', value: url.port });
    if (url.pathname && url.pathname !== '/') fields.push({ key: 'Path', value: url.pathname });
    if (url.search) fields.push({ key: 'Query', value: url.search });
    if (url.hash) fields.push({ key: 'Fragment', value: url.hash });
    return { type: 'url', label: 'URL / Web Link', icon: 'link', fields, notice: null };
  } catch {
    return { type: 'url', label: 'URL / Web Link', icon: 'link', fields: [{ key: 'URL', value: trimmed }], notice: null };
  }
}

function parseUpi(str) {
  if (!/^upi:\/\//i.test(str.trim())) return null;
  try {
    const url = new URL(str.trim());
    const p = url.searchParams;
    const fields = [
      p.get('pa') && { key: 'UPI ID (Payee)', value: p.get('pa') },
      p.get('pn') && { key: 'Payee Name', value: p.get('pn') },
      p.get('am') && { key: 'Amount', value: 'Rs. ' + p.get('am') },
      p.get('cu') && { key: 'Currency', value: p.get('cu') },
      p.get('tn') && { key: 'Transaction Note', value: p.get('tn') },
      p.get('tr') && { key: 'Transaction Ref', value: p.get('tr') },
    ].filter(Boolean);
    return { type: 'upi', label: 'UPI Payment', icon: 'payment', fields, notice: null };
  } catch { return null; }
}

function parseWifi(str) {
  if (!/^WIFI:/i.test(str.trim())) return null;
  const extract = (key) => { const m = str.match(new RegExp(key + ':([^;]*)')); return m ? m[1] : null; };
  const fields = [
    extract('S') && { key: 'Network Name (SSID)', value: extract('S') },
    extract('T') && { key: 'Security Type', value: extract('T') },
    extract('P') && { key: 'Password', value: extract('P'), sensitive: true },
    extract('H') && { key: 'Hidden Network', value: extract('H') === 'true' ? 'Yes' : 'No' },
  ].filter(Boolean);
  return { type: 'wifi', label: 'Wi-Fi Configuration', icon: 'wifi', fields, notice: null };
}

function parseVCard(str) {
  if (!/^BEGIN:VCARD/i.test(str.trim())) return null;
  const lines = str.split(/\r?\n/);
  const get = (prefix) => { const line = lines.find((l) => l.toLowerCase().startsWith(prefix.toLowerCase())); return line ? line.slice(prefix.length).trim() : null; };
  const fields = [
    get('FN:') && { key: 'Full Name', value: get('FN:') },
    get('ORG:') && { key: 'Organization', value: get('ORG:') },
    get('TITLE:') && { key: 'Title', value: get('TITLE:') },
    get('TEL:') && { key: 'Phone', value: get('TEL:') },
    get('EMAIL:') && { key: 'Email', value: get('EMAIL:') },
    get('URL:') && { key: 'Website', value: get('URL:') },
    get('BDAY:') && { key: 'Birthday', value: get('BDAY:') },
    get('NOTE:') && { key: 'Note', value: get('NOTE:') },
  ].filter(Boolean);
  return { type: 'vcard', label: 'Contact Card (vCard)', icon: 'contact', fields, notice: null };
}

function parseMeCard(str) {
  if (!/^MECARD:/i.test(str.trim())) return null;
  const extract = (key) => { const m = str.match(new RegExp(key + ':([^;]*)')); return m ? m[1] : null; };
  const fields = [
    extract('N') && { key: 'Name', value: extract('N').replace(/,/g, ' ') },
    extract('TEL') && { key: 'Phone', value: extract('TEL') },
    extract('EMAIL') && { key: 'Email', value: extract('EMAIL') },
    extract('ADR') && { key: 'Address', value: extract('ADR') },
    extract('URL') && { key: 'Website', value: extract('URL') },
    extract('BDAY') && { key: 'Birthday', value: extract('BDAY') },
    extract('NOTE') && { key: 'Note', value: extract('NOTE') },
  ].filter(Boolean);
  return { type: 'mecard', label: 'Contact Card (MeCard)', icon: 'contact', fields, notice: null };
}

function parseEmail(str) {
  if (!/^mailto:/i.test(str.trim())) return null;
  try {
    const url = new URL(str.trim());
    const p = url.searchParams;
    const fields = [
      { key: 'To', value: url.pathname },
      p.get('subject') && { key: 'Subject', value: p.get('subject') },
      p.get('body') && { key: 'Body', value: p.get('body') },
      p.get('cc') && { key: 'CC', value: p.get('cc') },
    ].filter(Boolean);
    return { type: 'email', label: 'Email (mailto)', icon: 'email', fields, notice: null };
  } catch { return null; }
}

function parsePhone(str) {
  if (!/^tel:/i.test(str.trim())) return null;
  return { type: 'phone', label: 'Phone Number', icon: 'phone', fields: [{ key: 'Number', value: str.trim().replace(/^tel:/i, '').trim() }], notice: null };
}

function parseSms(str) {
  if (!/^sms:/i.test(str.trim())) return null;
  try {
    const url = new URL(str.trim());
    const body = url.searchParams.get('body');
    const fields = [{ key: 'Number', value: url.pathname }, body && { key: 'Message', value: body }].filter(Boolean);
    return { type: 'sms', label: 'SMS Message', icon: 'sms', fields, notice: null };
  } catch { return null; }
}

function parseGeo(str) {
  if (!/^geo:/i.test(str.trim())) return null;
  const coords = str.trim().replace(/^geo:/i, '');
  const [latlon, rest] = coords.split('?');
  const [lat, lon, alt] = latlon.split(',');
  const fields = [
    lat && { key: 'Latitude', value: lat },
    lon && { key: 'Longitude', value: lon },
    alt && { key: 'Altitude', value: alt + ' m' },
    rest && { key: 'Query', value: rest },
  ].filter(Boolean);
  const mapUrl = lat && lon ? 'https://www.openstreetmap.org/?mlat=' + lat + '&mlon=' + lon : null;
  return { type: 'geo', label: 'Geographic Location', icon: 'geo', fields, mapUrl, notice: null };
}

function parseJson(str) {
  const trimmed = str.trim();
  if (!((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']')))) return null;
  try {
    const parsed = JSON.parse(trimmed);
    const fields = [];
    if (typeof parsed === 'object' && !Array.isArray(parsed)) {
      Object.entries(parsed).forEach(([k, v]) => fields.push({ key: k, value: typeof v === 'object' ? JSON.stringify(v) : String(v) }));
    } else if (Array.isArray(parsed)) {
      fields.push({ key: 'Array Length', value: String(parsed.length) });
      parsed.slice(0, 10).forEach((item, i) => fields.push({ key: '[' + i + ']', value: typeof item === 'object' ? JSON.stringify(item) : String(item) }));
    }
    return { type: 'json', label: 'JSON Data', icon: 'code', fields: fields.length > 0 ? fields : [{ key: 'Content', value: trimmed }], notice: null };
  } catch { return null; }
}

function parseXml(str) {
  const trimmed = str.trim();
  if (!trimmed.startsWith('<')) return null;
  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(trimmed, 'text/xml');
    if (doc.querySelector('parsererror')) return null;
    const root = doc.documentElement;
    const fields = Array.from(root.attributes).map((a) => ({ key: a.name, value: a.value }));
    Array.from(root.children).forEach((child) => { if (child.textContent.trim()) fields.push({ key: child.tagName, value: child.textContent.trim() }); });
    return { type: 'xml', label: 'XML Data <' + root.tagName + '>', icon: 'code', fields: fields.length > 0 ? fields : [{ key: 'Root Element', value: root.tagName }], notice: null };
  } catch { return null; }
}

export function parseQrPayload(data) {
  if (!data || !data.trim()) {
    return { type: 'empty', label: 'Empty Payload', icon: 'warning', fields: [], notice: 'The decoded QR payload is empty.' };
  }
  const trimmed = data.trim();
  return (
    (trimmed.includes(AADHAAR_OLD_QR_TAG) && parseAadhaarOldQr(trimmed)) ||
    detectAadhaarSecureQr(trimmed) ||
    parseUpi(trimmed) ||
    parseUrl(trimmed) ||
    parseWifi(trimmed) ||
    parseVCard(trimmed) ||
    parseMeCard(trimmed) ||
    parseEmail(trimmed) ||
    parsePhone(trimmed) ||
    parseSms(trimmed) ||
    parseGeo(trimmed) ||
    parseJson(trimmed) ||
    parseXml(trimmed) ||
    (/^\d+$/.test(trimmed) && { type: 'numeric', label: 'Numeric Data', icon: 'hash', fields: [{ key: 'Value', value: trimmed }, { key: 'Digit Count', value: String(trimmed.length) }], notice: null }) ||
    { type: 'text', label: 'Plain Text', icon: 'text', fields: [{ key: 'Content', value: trimmed }], notice: null }
  );
}

export default parseQrPayload;
