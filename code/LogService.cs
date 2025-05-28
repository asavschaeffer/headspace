using System;
using System.IO;
using UnityEngine; // Required for Time.time, Vector3, Application.persistentDataPath, Debug
using System.Collections.Generic;

/// <summary>
/// Static class responsible for efficiently logging game events to a binary file.
/// Uses a ring buffer for in-memory storage and flushes to disk periodically or on demand.
/// Designed for the LDAG system T0 prototype.
/// </summary>
public static class LogService
{
    // --- Event Type Constants (Renumbered & Categorized) ---

    // Categories (Using high nibble - allows 16 categories, 16 types per category)
    public const byte CAT_INPUT = 0x00; // 0
    public const byte CAT_PLAYER = 0x10; // 16
    public const byte CAT_WORLD = 0x20; // 32
    public const byte CAT_GAME = 0x30; // 48
    // Add CAT_COMBAT = 0x40, CAT_UI = 0x50 etc. later

    // Input Events (Category 0x00)
    public const byte TYPE_KEY_PRESS = CAT_INPUT | 0x01; // 1
    public const byte TYPE_KEY_RELEASE = CAT_INPUT | 0x02; // 2
    public const byte TYPE_CAMERA_ZOOM = CAT_INPUT | 0x03; // 3
    public const byte TYPE_MOUSE_POS_X = CAT_INPUT | 0x04; // 4
    public const byte TYPE_MOUSE_POS_Y = CAT_INPUT | 0x05; // 5

    // Player State/Movement Events (Category 0x10)
    public const byte TYPE_POSITION_X = CAT_PLAYER | 0x01; // 17
    public const byte TYPE_POSITION_Z = CAT_PLAYER | 0x02; // 18
    public const byte TYPE_JUMP_ACTION_STARTED = CAT_PLAYER | 0x04; // 20
    public const byte TYPE_SPRINT_ACTION_STARTED = CAT_PLAYER | 0x05; // 21
    public const byte TYPE_SPRINT_ACTION_ENDED = CAT_PLAYER | 0x06;   // 22
    public const byte TYPE_CROUCH_ACTION_STARTED = CAT_PLAYER | 0x07; // 23
    public const byte TYPE_CROUCH_ACTION_ENDED = CAT_PLAYER | 0x08;   // 24

    // World Interaction Events (Category 0x20)
    public const byte TYPE_COLLISION = CAT_WORLD | 0x01; // 33

    // Game State Events (Category 0x30)
    public const byte TYPE_CYCLE_CHANGE = CAT_GAME | 0x01; // 49

    // --- Reserved T1+ Placeholders --- (Assign category/number when implemented)
    // public const byte TYPE_TRAJECTORY = ???;
    // public const byte TYPE_TIME_DELTA = ???;
    // public const byte TYPE_STATIONARY_PERIOD = ???;

    // --- Configuration --- // (Rest of LogService remains the same)
    // ... private const BUFFER_CAPACITY ...
    // ... public const int ENTRY_SIZE ...
    // ... public const int HEADER_SIZE ...
    // ... public const int LOG_VERSION ...
    // ... etc ...
    // Reserved T1+ Event Types (placeholders for future expansion):
    public const byte TYPE_TRAJECTORY = 8;
    public const byte TYPE_TIME_DELTA = 9;
    public const byte TYPE_STATIONARY_PERIOD = 10;
    // Add more future types here...

    // --- Configuration ---
    private const int BUFFER_CAPACITY = 10000; // Number of log entries the ring buffer can hold
    public const int ENTRY_SIZE = 8;       // Bytes per log entry: 4B timestamp, 1B type, 3B data
    private const int LOG_VERSION = 1;          // Version of the log format
    public const int HEADER_SIZE = 4;          // Size of the version header

    // --- State Variables ---
    private static byte[] ringBuffer;           // The ring buffer storing log entries as bytes
    private static int head = 0;                // Index of the *next entry* to write (0 to BUFFER_CAPACITY-1)
    private static int tail = 0;                // Index of the *oldest entry* to flush (0 to BUFFER_CAPACITY-1)
    private static bool isFull = false;             // Indicates if the buffer has wrapped around completely
    private static string logFilePath;           // Full path to the log file
    private static readonly object bufferLock = new object(); // For thread safety
    private static Vector3 lastLoggedPosition = Vector3.zero; // Tracks the last logged player position
    private static float positionThreshold = 0.1f; // Minimum distance required to log a new position

    // --- Public Methods ---

    /// <summary>
    /// Initializes the logging system. Should be called once at game start.
    /// </summary>
    public static void Initialize()
    {
        // Reset state
        head = 0;
        tail = 0;
        isFull = false;
        lastLoggedPosition = Vector3.zero;

        // Set log file path
        logFilePath = Path.Combine(Application.persistentDataPath, "game_log.bin");

        // Initialize buffer
        ringBuffer = new byte[BUFFER_CAPACITY * ENTRY_SIZE];

        try
        {
            // Check file existence and handle version header
            if (!File.Exists(logFilePath))
            {
                // Create directory if it doesn't exist
                Directory.CreateDirectory(Path.GetDirectoryName(logFilePath));
                // Create file and write 4-byte version header
                using (FileStream fs = new FileStream(logFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    byte[] versionBytes = BitConverter.GetBytes(LOG_VERSION);
                    if (BitConverter.IsLittleEndian)
                    {
                        // Ensure consistent endianness if needed, though BitConverter usually matches system
                        // Array.Reverse(versionBytes); // Uncomment if a specific endianness is required
                    }
                    // Ensure we write exactly HEADER_SIZE bytes
                    byte[] headerBytes = new byte[HEADER_SIZE];
                    Buffer.BlockCopy(versionBytes, 0, headerBytes, 0, Math.Min(versionBytes.Length, HEADER_SIZE));
                    fs.Write(headerBytes, 0, HEADER_SIZE);
                }
                Debug.Log($"Log file created with version {LOG_VERSION}.");
            }
            else
            {
                // File exists, read header and verify version
                using (FileStream fs = new FileStream(logFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    if (fs.Length >= HEADER_SIZE)
                    {
                        byte[] headerBytes = new byte[HEADER_SIZE];
                        fs.Read(headerBytes, 0, HEADER_SIZE);
                        // if (BitConverter.IsLittleEndian) Array.Reverse(headerBytes); // Match potential reversal on write
                        int fileVersion = BitConverter.ToInt32(headerBytes, 0);
                        if (fileVersion != LOG_VERSION)
                        {
                            Debug.LogWarning($"Log file version mismatch! File version: {fileVersion}, Expected version: {LOG_VERSION}. Appending may lead to errors.");
                        }
                        else
                        {
                            Debug.Log($"Existing log file found with correct version {LOG_VERSION}.");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"Existing log file is smaller than header size ({fs.Length} bytes). Header potentially corrupt.");
                        // Decide recovery strategy: Overwrite? Append header? Throw error?
                        // For now, we'll just warn and allow appending past it.
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error during LogService initialization: {ex.Message}");
            // Consider disabling logging if init fails critically
        }

        Debug.Log("LogService Initialized. Logging to: " + logFilePath);
    }

    /// <summary>
    /// Logs a generic event with a 24-bit integer data payload.
    /// </summary>
    /// <param name="eventType">The type code for the event.</param>
    /// <param name="eventData">Integer data associated with the event (masked to 24 bits).</param>
    public static void LogEvent(byte eventType, int eventData)
    {
        // Using Time.time assuming Time.timeScale is handled appropriately (or switch to Time.unscaledTime if needed)
        float timestamp = Time.time;

        // Mask eventData to ensure it fits within 24 bits (3 bytes)
        eventData &= 0xFFFFFF;

        byte[] entryBytes = Pack8Bytes_413(timestamp, eventType, eventData);

        lock (bufferLock)
        {
            // Calculate the byte offset for the current head entry index
            int writeOffset = head * ENTRY_SIZE;

            // Write the 8 bytes to ringBuffer
            Buffer.BlockCopy(entryBytes, 0, ringBuffer, writeOffset, ENTRY_SIZE);

            // Advance head entry index
            head = (head + 1) % BUFFER_CAPACITY;

            if (isFull)
            {
                // Head caught up to tail after wrapping, buffer overflowed
                tail = (tail + 1) % BUFFER_CAPACITY; // Lose the oldest entry by advancing tail
                Debug.LogWarning("LogService: Buffer overflow, oldest log entry lost.");
                // Optional: Could trigger an immediate small flush here if desired,
                // but current design flushes only when becoming full or manually.
            }
            else if (head == tail)
            {
                // Head has just wrapped around to meet tail, buffer is now full
                isFull = true;
                // Flush immediately to prevent data loss on next write when full
                // NOTE: Calling internal flush assumes no recursive locking issues if FlushLog_Internal also locked,
                // but since FlushLog_Internal does IO, it's better called without holding the lock if possible,
                // or ensuring the lock is re-entrant. Current lock scope is okay.
                FlushLog_Internal(); // Flush the full buffer
                isFull = false; // Buffer is empty after flush
                tail = head;   // Reset tail to head after flush
            }
        }
    }

    /// <summary>
    /// Manually triggers a flush of the log buffer to disk.
    /// </summary>
    public static void FlushLog()
    {
        lock (bufferLock)
        {
            FlushLog_Internal();
        }
    }

    /// <summary>
    /// Logs the player's position, but only if it has changed significantly since the last log.
    /// Logs X and Z coordinates as two separate events.
    /// </summary>
    /// <param name="x">Current X coordinate.</param>
    /// <param name="z">Current Z coordinate.</param>
    public static void LogPositionEvent(float x, float z)
    {
        Vector3 currentPosition = new Vector3(x, 0, z); // Ignore Y for T0 distance check

        // Check distance using squared magnitude for efficiency
        if ((currentPosition - lastLoggedPosition).sqrMagnitude > positionThreshold * positionThreshold)
        {
            // Scale position coordinates to integers for 24-bit packing
            // Scaling by 1000 gives precision up to 0.001 units
            int scaledX = (int)(x * 1000.0f);
            int scaledZ = (int)(z * 1000.0f);

            // Lock once to ensure X and Z events are logged consecutively if possible
            lock (bufferLock)
            {
                LogEvent(TYPE_POSITION_X, scaledX);
                LogEvent(TYPE_POSITION_Z, scaledZ); // Note: This LogEvent call will acquire the lock again - okay as C# locks are re-entrant
                lastLoggedPosition = currentPosition; // Update only if logged
            }
        }
    }

    // --- Specific Helper Methods ---

    public static void LogKeyPressEvent(KeyCode key)
    {
        LogEvent(TYPE_KEY_PRESS, (int)key);
    }

    public static void LogKeyReleaseEvent(KeyCode key)
    {
        LogEvent(TYPE_KEY_RELEASE, (int)key);
    }

    public static void LogJumpEvent()
    {
        LogEvent(TYPE_JUMP_ACTION_STARTED, 0); // Data payload is not used for jump event
    }

    public static void LogCollisionEvent(int objectID)
    {
        LogEvent(TYPE_COLLISION, objectID);
    }

    public static void LogCycleChangeEvent(int cycleType)
    {
        // Example: 0 for day, 1 for night
        LogEvent(TYPE_CYCLE_CHANGE, cycleType);
    }

    /// <summary>
    /// Gets a string representing the current state of the buffer (for debugging).
    /// </summary>
    /// <returns>Status string.</returns>
    public static string GetBufferStats()
    {
        lock (bufferLock)
        {
            // Calculate number of entries currently holding valid data to be flushed
            int entryCount;
            if (head == tail)
            {
                entryCount = isFull ? BUFFER_CAPACITY : 0;
            }
            else if (head > tail)
            {
                entryCount = head - tail;
            }
            else
            { // head < tail (wrapped)
                entryCount = BUFFER_CAPACITY - tail + head;
            }
            return $"LogService Stats: Head={head}, Tail={tail}, IsFull={isFull}, Count={entryCount}/{BUFFER_CAPACITY}";
        }
    }

    /// <summary>
    /// Returns a list of LogEntry objects representing the current contents of the ring buffer
    /// without modifying the buffer state.
    /// </summary>
    /// <returns>List of LogEntry objects from the current buffer, or empty list if buffer is empty.</returns>
    public static List<LogEntry> GetCurrentBufferEntries()
    {
        List<LogEntry> entries = new List<LogEntry>();
        byte[] entryBuffer = new byte[ENTRY_SIZE]; // Temp buffer for one entry

        lock (bufferLock) // Ensure thread safety while reading buffer state
        {
            int currentHead = head;
            int currentTail = tail;
            bool bufferWasFull = isFull;

            int entryCount;
            // Calculate number of valid entries currently in the buffer
            if (currentHead == currentTail) {
                entryCount = bufferWasFull ? BUFFER_CAPACITY : 0;
            } else if (currentHead > currentTail) {
                entryCount = currentHead - currentTail;
            } else { // currentHead < currentTail (wrapped)
                entryCount = BUFFER_CAPACITY - currentTail + currentHead;
            }

            if (entryCount == 0) { return entries; }

            entries = new List<LogEntry>(entryCount); // Pre-allocate
            int readIndex = currentTail;

            // Iterate through the valid entries
            for (int i = 0; i < entryCount; i++)
            {
                int entryByteOffset = readIndex * ENTRY_SIZE;
                try {
                    Buffer.BlockCopy(ringBuffer, entryByteOffset, entryBuffer, 0, ENTRY_SIZE);
                    var unpacked = Unpack8Bytes(entryBuffer);
                    entries.Add(new LogEntry(unpacked.timestamp, unpacked.eventType, unpacked.eventData));
                } catch (Exception ex) {
                    Debug.LogError($"LogService.GetCurrentBufferEntries: Error at index {readIndex}: {ex.Message}");
                }
                readIndex = (readIndex + 1) % BUFFER_CAPACITY; // Move to next index
            }
        } // End lock
        return entries;
    }

    // --- Private Helper Methods ---

    /// <summary>
    /// Internal implementation of flushing the buffer to disk. Assumes lock is held.
    /// </summary>
    private static void FlushLog_Internal()
    {
        int currentTail = tail; // Capture state at start of flush
        int currentHead = head;
        bool bufferWasFull = isFull;

        if (currentHead == currentTail && !bufferWasFull)
        {
            // Buffer is empty
            return;
        }

        try
        {
            // Use FileStream for efficient binary append.
            // FileShare.None prevents other processes from accessing the file during write.
            using (FileStream fileStream = new FileStream(logFilePath, FileMode.Append, FileAccess.Write, FileShare.None))
            {
                if (!bufferWasFull && currentHead > currentTail)
                {
                    // Simple case: Data is contiguous from tail to head
                    int startByte = currentTail * ENTRY_SIZE;
                    int countBytes = (currentHead - currentTail) * ENTRY_SIZE;
                    fileStream.Write(ringBuffer, startByte, countBytes);
                }
                else // Buffer wrapped around or was full
                {
                    // Write from tail to the end of the buffer array
                    int startByteTail = currentTail * ENTRY_SIZE;
                    int countBytesTail = (BUFFER_CAPACITY * ENTRY_SIZE) - startByteTail;
                    fileStream.Write(ringBuffer, startByteTail, countBytesTail);

                    // Write from the beginning of the buffer array to the head
                    if (currentHead > 0)
                    {
                        int countBytesHead = currentHead * ENTRY_SIZE;
                        fileStream.Write(ringBuffer, 0, countBytesHead);
                    }
                }
            }

            // Update tail to reflect flushed data ONLY AFTER successful write
            tail = currentHead;
            isFull = false; // Buffer is effectively empty after flushing all content
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to flush game log: {ex.Message}\n{ex.StackTrace}");
            // Consider: What happens if flush fails? Retry? Lose data? Stop logging?
            // For T0, we log the error and potentially lose the buffered data.
            // Tail is NOT updated, so next flush will try again (potentially duplicating data if partially written).
        }
    }

    /// <summary>
    /// Packs timestamp, event type, and 24-bit event data into an 8-byte array.
    /// Format: [4B:float:Timestamp][1B:byte:Type][3B:int24:Data]
    /// </summary>
    private static byte[] Pack8Bytes_413(float timestamp, byte eventType, int eventData)
    {
        byte[] packed = new byte[ENTRY_SIZE]; // Should be 8
        byte[] timeBytes = BitConverter.GetBytes(timestamp);
        byte[] dataBytes = BitConverter.GetBytes(eventData); // We'll only use the first 3 bytes

        // Ensure consistent endianness if targeting multiple platforms with different defaults
        // Although BitConverter usually uses system endianness, explicit checks might be needed for cross-platform save files
        // if (!BitConverter.IsLittleEndian) { /* Reverse bytes if needed */ }

        // Copy timestamp (Bytes 0-3)
        Buffer.BlockCopy(timeBytes, 0, packed, 0, 4);

        // Copy event type (Byte 4)
        packed[4] = eventType;

        // Copy first 3 bytes of data (Bytes 5-7, assumes little-endian for dataBytes)
        packed[5] = dataBytes[0];
        packed[6] = dataBytes[1];
        packed[7] = dataBytes[2];

        return packed;
    }

    /// <summary>
    /// Unpacks an 8-byte log entry back into its components. Public for use by analysis service.
    /// Format: [4B:float:Timestamp][1B:byte:Type][3B:int24:Data]
    /// </summary>
    public static (float timestamp, byte eventType, int eventData) Unpack8Bytes(byte[] entry)
    {
        if (entry == null || entry.Length != ENTRY_SIZE)
        {
            throw new ArgumentException($"Invalid log entry size. Expected {ENTRY_SIZE} bytes.");
        }

        // Ensure consistent endianness on read if needed
        // if (!BitConverter.IsLittleEndian) { /* Reverse bytes if needed */ }

        float timestamp = BitConverter.ToSingle(entry, 0); // Bytes 0-3
        byte eventType = entry[4];                         // Byte 4

        // Reconstruct 24-bit integer from bytes 5, 6, 7 (assuming little-endian)
        // Pad with a zero byte at the end to make it 32-bit for BitConverter.ToInt32
        // Note: This correctly handles positive numbers up to 0xFFFFFF. Negative numbers in 24-bit two's complement wouldn't be preserved correctly this way.
        // Assuming eventData is primarily IDs or scaled positive values for T0.
        int eventData = entry[5] | (entry[6] << 8) | (entry[7] << 16);

        return (timestamp, eventType, eventData);
    }

    /// <summary>
    /// Helper method to unpack a scaled position coordinate from the 24-bit integer data. Public for analysis.
    /// </summary>
    /// <param name="packedValue">The 24-bit integer containing the scaled coordinate.</param>
    /// <returns>The float coordinate.</returns>
    public static float UnpackPositionCoordinate(int packedValue)
    {
        // Reverse the scaling applied during logging
        // Note: Need to handle potential sign if the 24th bit was used for sign representation,
        // but current packing assumes positive scaled values.
        // If values could truly be negative in the original float, scaling might need adjustment
        // or sign bit needs explicit handling during unpack. For T0, assume positive world coords.
        return (float)packedValue / 1000.0f;
    }


    // --- Placeholder methods for T1+ features ---

    /// <summary>
    /// T1+: Placeholder for logging trajectory data.
    /// </summary>
    public static void LogTrajectory(Vector3[] points)
    {
        // TODO: Implement trajectory logging for T1+.
        // Could involve multiple events or a dedicated complex event type.
    }

    /// <summary>
    /// T1+: Placeholder for logging time deltas (for variable rate logging).
    /// </summary>
    private static void LogTimeDelta(float delta)
    {
        // TODO: Implement time delta logging for T1+.
        // Likely involves a specific event type (e.g., TYPE_TIME_DELTA).
    }

    /// <summary>
    /// T1+: Placeholder for logging extended stationary periods efficiently.
    /// </summary>
    private static void LogStationaryPeriod(float duration)
    {
        // TODO: Implement stationary period logging for T1+.
        // Likely involves a specific event type (e.g., TYPE_STATIONARY_PERIOD).
    }
}