# MANUS Glove Ergonomics Data 핵심 정리

## 1. 데이터 구조 (ROS2 토픽 기준)

ROS2 토픽 `manus_glove_#` → `ManusGlove.ergonomics[]` 배열
- 각 원소는 `ManusErgonomics{ type: string, value: float32 }`
- 값은 **degree(°)** 단위, 관절 각도
- 좌/우 글러브는 토픽이 따로 분리되며, `ManusGlove.side` ("Left" 또는 "Right")로 구분
- ROS2에서 `type` 문자열은 좌우 동일 (예: 좌/우 모두 `"IndexMCPStretch"`) — **side 필드로 좌우 판별 필수**

## 2. 채널 구성 (한 손 20개)

| 손가락 | 채널 (ROS2 string) | 해부학적 의미 |
|---|---|---|
| Thumb | `ThumbMCPSpread` | CMC abduction (palmar abduction) |
| Thumb | `ThumbMCPStretch` | CMC flexion |
| Thumb | `ThumbPIPStretch` | MCP flexion |
| Thumb | `ThumbDIPStretch` | IP flexion |
| Index/Middle/Ring/Pinky | `{Finger}Spread` | MCP abduction/adduction |
| Index/Middle/Ring/Pinky | `{Finger}MCPStretch` | MCP flexion |
| Index/Middle/Ring/Pinky | `{Finger}PIPStretch` | PIP flexion |
| Index/Middle/Ring/Pinky | `{Finger}DIPStretch` | DIP flexion |

> ⚠️ SDK 명명상 "Stretch"는 실제로 **flexion(굴곡)** 각도입니다. CSV exporter는 `*_Flex`로 표기하는 것과 같은 의미.

## 3. 부호 규약 (로봇 매핑용 핵심)

### Flex 계열 (`*Stretch`) — 모든 손가락 공통, 좌우 동일
| 부호 | 사람 손 동작 | 로봇 관절 매핑 |
|---|---|---|
| **+** | **Flexion** (손바닥 쪽으로 굽힘) | 굽힘 방향을 + 축으로 정렬 |
| **−** | (Hyper)extension (손등 쪽으로 젖힘) | 반대 방향 (보통 무시 가능, 로봇이 hyperextension 못 함) |
| 0 | 손가락 직선 | 중립 |

### Finger Spread (`{Index/Middle/Ring/Pinky}Spread`) — **좌우 mirror 가능성**
| 부호 | 사람 손 동작 | 주의 |
|---|---|---|
| **+** | 한 방향 splay | 좌/우손에 따라 radial/ulnar이 mirror됨 |
| **−** | 반대 방향 splay | 실제 측정으로 확인 필요 |
| 0 | Hand forward vector 방향 | 중립 |

> 우손 CSV 데이터 패턴: Index spread는 vector spread 시 + 방향, Pinky spread는 − 방향으로 큰 값 → **Index = radial(+), Pinky = ulnar(−)**일 가능성이 높음. 좌손은 mirror.

### Thumb Spread (`ThumbMCPSpread`)
| 부호 | 사람 손 동작 | 로봇 매핑 |
|---|---|---|
| **+** | Palmar abduction (엄지가 손바닥 평면에서 멀어짐) | 보통 + 축 정렬 |
| **−** | Adduction (엄지가 검지 쪽 손바닥에 붙음) | − |
| 0 | 손바닥 평면 | 중립 |

### Thumb Flex (`ThumbMCPStretch` / `PIPStretch` / `DIPStretch`)
| 부호 | 동작 |
|---|---|
| **+** | Flexion (손바닥을 가로지르는 방향) |
| **−** | Extension |

## 4. 로봇 매핑 시 체크리스트

1. **부호 정렬**: 로봇 관절의 + 회전 방향이 사람 손의 flexion(+)과 일치하는지 확인 → 안 맞으면 `-1` 곱하기
2. **좌우 mirror 처리**: Spread 채널은 좌/우손 글러브 캘리브레이션 후 직접 부호 검증 권장. 좌손 데이터를 우손 로봇에 적용하면 spread가 반대로 갈 수 있음
3. **각도 범위 클리핑**: 사람은 hyperextension(−) 가능하지만 대부분 로봇 손은 불가 → `max(0, value)` 처리 후 매핑
4. **단위 변환**: degree → radian (`* π/180`)
5. **DIP-PIP 종속 관계**: 사람 손은 PIP/DIP가 거의 1:0.67 비율로 함께 굽힘 (anatomically coupled). 로봇이 underactuated면 PIP만 사용하고 DIP는 무시하거나 결합 모델 적용

## 5. 검증용 캘리브레이션 자세

| 자세 | 예상 부호 변화 |
|---|---|
| T-pose (손 펴기) | 모두 ≈ 0 |
| 주먹 쥐기 | 모든 `*Stretch` 큰 양수 (+) |
| 손가락 쫙 벌리기 | `IndexSpread` 와 `PinkySpread` 가 서로 반대 부호로 극값 |
| 엄지 들어올리기 (palm에서 멀리) | `ThumbMCPSpread` 큰 양수 (+) |
| 엄지를 손바닥 가로질러 굽히기 | `ThumbMCPStretch` 양수 (+) |

> 좌/우 글러브 각각 위 자세를 수행하면서 ROS2 토픽 값을 캡처하면 부호 매핑이 한 번에 결정됩니다.

## 6. 명심할 한 줄

> **"Stretch는 실제로 Flexion이고, +가 굽힘이다. Spread는 좌/우손에서 부호 방향이 반대일 수 있으니 직접 측정해서 확인할 것."**
