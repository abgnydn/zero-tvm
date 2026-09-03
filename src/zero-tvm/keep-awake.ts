/**
 * KEEP-AWAKE — screen wake-lock plus a silent audio track, for a tab that serves.
 *
 * The wake lock stops the machine sleeping; the audio track is the standard
 * exemption from background-tab throttling (measured: a backgrounded host
 * generated at ~23 tok/s where the focused tab does ~65, and served weights at
 * ~1 MB/s). Honestly labeled: the browser shows its audio indicator on the tab.
 *
 * ONE generator, both hosting surfaces. share.html's host and helper roles have
 * carried this since they shipped. The entrance's ⟁ Room tool — the PRIMARY
 * hosting path now that the swarm's first machine boots in place rather than
 * opening share.html — took a best-effort `wakeLock.request('screen')` and
 * nothing else: no audio track, so no exemption, so the flow this site
 * recommends was the throttled one, with no control to fix it, while the path
 * it demoted had one. Copying the wiring across would have made "how a tab
 * escapes background throttling" a hand-copied fact, which is the thing that
 * drifts, so it lives here and both surfaces call it.
 *
 * IMPORTS NOTHING, deliberately. share.html's guest role needs no WebGPU and
 * downloads nothing, so no module on its path may reach the loader chain; the
 * entrance's room reaches model-select → weight-loader and cannot be on it.
 * This file touches `document`, `navigator.wakeLock` and `AudioContext`, and
 * nothing else, so either surface can take it.
 *
 * The audio never starts on its own. It starts on the operator's click — which
 * is also the user gesture an AudioContext needs in order to leave `suspended`.
 * A room opening is not a gesture and must not start a track.
 */

/** What the control does, in one sentence, in one place: share.html prints it
 *  under the button as a `.sw-note`, the entrance carries it on the button
 *  itself, the way the ⟁ Room tool beside it explains its own job. */
export const KEEP_AWAKE_NOTE = 'Screen wake-lock plus a silent audio track, so the browser throttles '
  + 'generation and weight serving less while this tab is in the background. The tab shows its audio indicator.'

/**
 * Wire a button as the keep-awake toggle: `aria-pressed` for the semantics,
 * `.cs-tool-live` for the lit state the scene's other tools use.
 *
 * Takes the element rather than looking one up, because the two surfaces build
 * it differently — share.html writes a `.cs-chat-tool` row into its sheet, the
 * entrance creates one in the chat head next to ⟁ Room — and an id lookup here
 * would put a third fact (the id) in a place neither of them owns.
 */
export function wireKeepAwake(btn: HTMLElement | null): void {
  if (!btn) return
  let on = false
  let wakeLock: WakeLockSentinel | null = null
  let audioCtx: AudioContext | null = null
  // WHAT THE BUTTON SAYS IS WHAT THE MACHINE IS DOING, and `apply` has an
  // await in the middle of making that true. Each call takes the next
  // generation; after the await, a generation that has moved means the state
  // this work was requested for is gone. Then the sentinel it just took is
  // handed back rather than installed, and no audio graph is built at all.
  //
  // Without this, two clicks in one tick interleave — the OFF branch runs to
  // completion while both holds are still null, so both releases are no-ops,
  // and the ON branch's await then installs a live sentinel and a running
  // context underneath a control that already reads off. The screen stays
  // awake, the tab stays exempt from throttling, and the audio indicator this
  // file offers as its proof of honesty becomes the only true thing on screen.
  //
  // The alternative — serialising the toggles behind a queue — would make the
  // button either lag the operator's clicks or appear to respond while doing
  // nothing, and this control's whole job is to not lie about the machine.
  // Comparing the resolved `want` against `on` is NOT enough either: on
  // off→on→off→on the first request comes back to a state that matches it
  // again, and installing it there leaks the sentinel the second one takes.
  let generation = 0
  btn.setAttribute('aria-pressed', 'false')
  const apply = async (want: boolean): Promise<void> => {
    const mine = ++generation
    if (want) {
      let taken: WakeLockSentinel | null = null
      try { taken = await navigator.wakeLock.request('screen') } catch { /* unsupported / not visible */ }
      if (mine !== generation) {
        // Nobody wants this any more — including the grant that arrived after
        // the operator gave up on it.
        void taken?.release().catch(() => {})
        return
      }
      if (taken) wakeLock = taken
      if (!audioCtx) {
        audioCtx = new AudioContext()
        const osc = audioCtx.createOscillator()
        const gain = audioCtx.createGain()
        gain.gain.value = 0.0001   // inaudible, but "playing" as far as the scheduler cares
        osc.connect(gain).connect(audioCtx.destination)
        osc.start()
      }
      void audioCtx.resume()
    } else {
      void wakeLock?.release().catch(() => {})
      wakeLock = null
      void audioCtx?.suspend()
    }
  }
  btn.addEventListener('click', () => {
    on = !on
    btn.setAttribute('aria-pressed', String(on))
    btn.classList.toggle('cs-tool-live', on)
    void apply(on)
  })
  document.addEventListener('visibilitychange', () => {
    // The UA releases wake locks on hide; re-acquire when we come back.
    if (on && document.visibilityState === 'visible') void apply(true)
  })
}
