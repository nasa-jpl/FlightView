# FlightView Build Options

## macOS

**Default build (Metal GPU + RTP):**
```bash
./build_with_metal.sh
```

**Note:** macOS automatically uses Metal for GPU acceleration. RTP over network is the only supported camera interface.

---

## Linux

### Default Build (CUDA GPU + RTP)
```bash
cd backend && make clean && make
cd .. && qmake && make clean && make
```

### Build Options

Linux supports two independent build options via Makefile variables:

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `USE_CUDA` | 0 or 1 | 1 | NVIDIA CUDA GPU acceleration |
| `USE_CAMERALINK` | 0 or 1 | 0 | Camera Link interface |

These options are **independent** and can be combined in any way.

---

### GPU Acceleration

**CUDA (default):**
```bash
cd backend && make clean && make
cd .. && qmake && make clean && make
```

**CPU-only (specify USE_CUDA=0):**
```bash
cd backend && make clean && make USE_CUDA=0
cd .. && qmake CONFIG+=nocuda && make clean && make
```

**Requirements for CUDA:**
- NVIDIA GPU with CUDA support
- CUDA toolkit in `/usr/local/cuda`
- `nvcc` compiler available

---

### Camera Interface

**RTP (default):**
```bash
cd backend && make clean && make
```

**Camera Link (specify USE_CAMERALINK=1):**
```bash
cd backend && make clean && make USE_CAMERALINK=1
```

**Requirements for Camera Link:**
- Camera Link frame grabber hardware
- EDT SDK installed
- EDT libraries in `/opt/EDTpdv`

---

### Combined Examples

| GPU | Camera | Build Command |
|-----|--------|---------------|
| CUDA | RTP | `make` (default) |
| CPU | RTP | `make USE_CUDA=0` |
| CUDA | Camera Link | `make USE_CAMERALINK=1` |
| CPU | Camera Link | `make USE_CUDA=0 USE_CAMERALINK=1` |

**Full build examples:**

**CUDA + Camera Link:**
```bash
cd backend && make clean && make USE_CAMERALINK=1
cd .. && qmake && make clean && make
```

**CPU + Camera Link:**
```bash
cd backend && make clean && make USE_CUDA=0 USE_CAMERALINK=1
cd .. && qmake CONFIG+=nocuda && make clean && make
```

---

## Interface Comparison

| Feature | RTP | Camera Link |
|---------|------|-------------|
| Bandwidth | Network-dependent | ~850 MB/s |
| Transport | Network (Ethernet, WiFi, localhost) | Direct frame grabber |
| Hardware | Network interface | Frame grabber card |
| Use Case | Flexible networking | High-speed direct capture |

---

## Troubleshooting

**"Built without CUDA support"**
- Rebuild both backend and frontend with CUDA enabled (default on Linux)

**"invalid device ordinal"**
- Check GPU with `nvidia-smi` or build with `USE_CUDA=0`

**EDT library errors**
- Install EDT SDK and set: `export LD_LIBRARY_PATH=/opt/EDTpdv:$LD_LIBRARY_PATH`

**Switching options**
- Always clean before changing options: `cd backend && make clean && cd .. && make clean`

---

*Last Updated: January 2026*
