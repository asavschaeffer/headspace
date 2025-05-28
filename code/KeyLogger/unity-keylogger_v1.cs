using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel; // Required for KeyboardEventPtr

public class RawInputLogger : MonoBehaviour
{
    void OnEnable()
    {
        // Subscribe to the global event stream for all input devices
        InputSystem.onEvent += OnInputSystemEvent;
    }

    void OnDisable()
    {
        // Unsubscribe when this object is disabled or destroyed
        InputSystem.onEvent -= OnInputSystemEvent;
    }

    private void OnInputSystemEvent(InputEventPtr eventPtr, InputDevice device)
    {
        // We are only interested in events from keyboard devices
        if (device is Keyboard keyboard)
        {
            // Check if the event is a state event (which includes key presses/releases)
            if (eventPtr.IsA<StateEvent>() || eventPtr.IsA<DeltaStateEvent>())
            {
                // Iterate through all changed controls in this event
                foreach (var control in eventPtr.GetControls())
                {
                    // Check if the control is a key and its state changed
                    if (control is KeyControl keyControl)
                    {
                        // WasPressedThisFrame() and WasReleasedThisFrame() check the *current* frame's change.
                        // For onEvent, we need to check the state change within the event itself.
                        // We can do this by comparing the current value to the previous value if available,
                        // or more simply, check the value if it's a button-like press.

                        // A simpler way to check for press/release in onEvent for KeyControl:
                        // keyControl.ReadValueFromEvent(eventPtr) gives the value of the key in this event.
                        // For simple keys, it's 1.0f if pressed, 0.0f if released.
                        // To avoid logging every state check, we only care about the transition.

                        // The KeyboardEventPtr is more direct for KeyDown/KeyUp
                        if (KeyboardEvent.current != null && KeyboardEvent.current.Equals(eventPtr))
                        {
                            var keyEvent = KeyboardEvent.current;
                            if (keyEvent.isPressed) // This is effectively KeyDown
                            {
                                //UnityEngine.Debug.Log($"Raw Key Down: {keyControl.keyCode}");
                                LogService.LogEvent(LogService.TYPE_RAW_KEY_DOWN, (int)keyControl.keyCode);
                            }
                            else // This is effectively KeyUp (when isPressed is false for a processed event)
                            {
                                //UnityEngine.Debug.Log($"Raw Key Up: {keyControl.keyCode}");
                                LogService.LogEvent(LogService.TYPE_RAW_KEY_UP, (int)keyControl.keyCode);
                            }
                        }
                    }
                }
            }
        }
    }
}