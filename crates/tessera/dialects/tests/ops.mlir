func.func @g(%x: tensor<1x6x6x4xi8>, %w: tensor<4x3x3x4xi8>, %b: tensor<4xi32>, %m: tensor<4xi32>) -> tensor<1x12x12x4xi8> {
  %0 = "tessera.conv"(%x, %w, %b, %m) {kind = "conv2d", name = "stem", index = 0 : i64, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = 1 : i64, reflect = true, groups = 1 : i64, shift = 22 : i64, clamp = array<i64: 0, 127>, multiplier = array<i64: 1, 2, 3, 4>} : (tensor<1x6x6x4xi8>, tensor<4x3x3x4xi8>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x6x6x4xi8>
  %1 = "tessera.add"(%x, %0) {index = 1 : i64, multiplier = array<i64: 5>, shift = 3 : i64, clamp = array<i64: -127, 127>, range = array<i64: -254, 254>} : (tensor<1x6x6x4xi8>, tensor<1x6x6x4xi8>) -> tensor<1x6x6x4xi8>
  %2 = "tessera.upsample"(%1) {index = 2 : i64, scale = 2 : i64, range = array<i64: -254, 254>} : (tensor<1x6x6x4xi8>) -> tensor<1x12x12x4xi8>
  "tessera.engine"() {sym_name = "e0", mode = "shared", simd = 4 : i64, pe = 4 : i64, vectors = 1 : i64} : () -> ()
  "tessera.bind"() {op = 0 : i64, engine = "e0", segment = 0 : i64, step = 0 : i64} : () -> ()
  "tessera.bind"() {op = 1 : i64, engine = "e0", segment = 0 : i64, step = 1 : i64} : () -> ()
  "tessera.bind"() {op = 2 : i64, engine = "e0", segment = 0 : i64, step = 2 : i64} : () -> ()
  "tessera.buffer"() {producer = 0 : i64, residence = "kbank", banks = 2 : i64, base = 0 : i64, words = 36 : i64, first_step = 0 : i64, last_step = 1 : i64} : () -> ()
  "tessera.segment"() {index = 0 : i64, mode = "shared", ops = array<i64: 0, 1, 2>, engines = ["e0"], cycles = 100 : i64} : () -> ()
  "tessera.schedule"() {segments = array<i64: 0>, initiation_interval = 100 : i64, latency = 100 : i64} : () -> ()
  return %2 : tensor<1x12x12x4xi8>
}
