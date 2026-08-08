import Foundation
import CoreMIDI

// MARK: - MidiEngine
//
// Reads MIDI from a source port (the sn2 virtual bus by default) and feeds a
// compact, normalised state to the visuals. The Novation/sn2 rig sends no MIDI
// clock — instead the ch-16 Program Change is the sync anchor (t≡0), and every
// modulator is a smooth sine. We surface the live CCs/notes plus that anchor
// phase so the mixer can lock synced oscillators to it.
//
// Connect it to a source whose name contains MIDI_PORT (default "sn2"/"visuals").

final class MidiEngine {

    /// Opt-in: set MIDI=1 (or MIDI_PORT=<name>) to enable MIDI reactivity.
    static let enabled = ProcessInfo.processInfo.environment["MIDI"] == "1"
                      || ProcessInfo.processInfo.environment["MIDI_PORT"] != nil

    /// Called ~60×/s on the main thread with the current MIDI state.
    var onUpdate: (([String: Any]) -> Void)?

    private var client = MIDIClientRef()
    private var port   = MIDIPortRef()
    private var connected = false
    private var timer: Timer?

    private let lock = NSLock()
    private var cc   = [[UInt8]](repeating: [UInt8](repeating: 0, count: 128), count: 16)
    private var bend = [Double](repeating: 0, count: 16)     // −1..1 per channel
    private var active = Set<Int>()                          // (ch<<8)|note
    private var lastNote = 60.0, lastVel = 0.0
    private var hits = 0                                     // note-on counter → pulse edge
    private var anchor: CFAbsoluteTime = 0                   // time of last ch-16 PC
    private var program = -1

    private let wantName: String

    init() {
        wantName = ProcessInfo.processInfo.environment["MIDI_PORT"] ?? "sn2"
    }

    func start() {
        guard MIDIClientCreateWithBlock("ButterchurnMIDI" as CFString, &client, nil) == noErr else {
            print("[MIDI] client create failed"); return
        }
        let refcon = Unmanaged.passUnretained(self).toOpaque()
        guard MIDIInputPortCreate(client, "in" as CFString, Self.readProc, refcon, &port) == noErr else {
            print("[MIDI] input port create failed"); return
        }
        connectSources()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { [weak self] _ in self?.emit() }
    }

    private func connectSources() {
        let n = MIDIGetNumberOfSources()
        var matched = 0
        var names: [String] = []
        for i in 0..<n {
            let src = MIDIGetSource(i)
            let name = Self.name(of: src)
            names.append(name)
            if name.lowercased().contains(wantName.lowercased()) ||
               name.lowercased().contains("visuals") {
                if MIDIPortConnectSource(port, src, nil) == noErr { matched += 1; print("[MIDI] connected: \(name)") }
            }
        }
        connected = matched > 0
        if !connected { print("[MIDI] no source matching '\(wantName)'. Sources: \(names)") }
    }

    // MARK: read callback (C convention, MIDI thread)

    private static let readProc: MIDIReadProc = { pktList, refCon, _ in
        guard let refCon = refCon else { return }
        let engine = Unmanaged<MidiEngine>.fromOpaque(refCon).takeUnretainedValue()
        var pkt = pktList.pointee.packet
        for _ in 0..<pktList.pointee.numPackets {
            let len = Int(pkt.length)
            withUnsafeBytes(of: pkt.data) { buf in engine.parse(buf, len) }
            pkt = MIDIPacketNext(&pkt).pointee
        }
    }

    private func parse(_ b: UnsafeRawBufferPointer, _ len: Int) {
        var i = 0
        lock.lock(); defer { lock.unlock() }
        while i < len {
            let status = b[i]
            if status < 0x80 { i += 1; continue }
            let type = status & 0xF0, ch = Int(status & 0x0F)
            switch type {
            case 0x90 where i+2 < len:  // note on
                let note = Int(b[i+1]), vel = Int(b[i+2])
                if vel > 0 { active.insert((ch<<8)|note); lastNote = Double(note); lastVel = Double(vel)/127; hits += 1 }
                else { active.remove((ch<<8)|note) }
                i += 3
            case 0x80 where i+2 < len:  // note off
                active.remove((ch<<8)|Int(b[i+1])); i += 3
            case 0xB0 where i+2 < len:  // control change
                cc[ch][Int(b[i+1]) & 0x7F] = b[i+2]; i += 3
            case 0xE0 where i+2 < len:  // pitch bend
                bend[ch] = (Double((Int(b[i+2])<<7)|Int(b[i+1])) - 8192) / 8192; i += 3
            case 0xC0 where i+1 < len:  // program change — ch 16 (index 15) is the anchor
                if ch == 15 { anchor = CFAbsoluteTimeGetCurrent(); program = Int(b[i+1]) }
                i += 2
            default: i += 1
            }
        }
    }

    // MARK: emit compact state to the visuals

    private func emit() {
        lock.lock()
        func avg(_ c: Int) -> Double { var s = 0.0; for ch in 0..<8 { s += Double(cc[ch][c]) }; return s / 8 / 127 }
        let cutoff = avg(74), reso = avg(71), hard = avg(83), sync = avg(19), mod = avg(1)
        var b = 0.0; for ch in 0..<8 { b += bend[ch] }; b /= 8
        let notes = active.count, note = lastNote, vel = lastVel, hitCount = hits
        let anch = anchor, prog = program, conn = connected
        lock.unlock()
        let beat = anch > 0 ? (CFAbsoluteTimeGetCurrent() - anch) : -1   // seconds since anchor
        onUpdate?([
            "cutoff": cutoff, "reso": reso, "hard": hard, "sync": sync, "mod": mod,
            "bend": b, "notes": notes, "note": note, "vel": vel, "hits": hitCount,
            "beat": beat, "program": prog, "connected": conn,
        ])
    }

    static func name(of ep: MIDIEndpointRef) -> String {
        var cf: Unmanaged<CFString>?
        MIDIObjectGetStringProperty(ep, kMIDIPropertyDisplayName, &cf)
        return (cf?.takeRetainedValue()) as String? ?? "?"
    }
}
