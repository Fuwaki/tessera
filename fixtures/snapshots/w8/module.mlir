module {
  func.func @tessera_graph(%input: tensor<1x6x6x3xi8>, %stem__filter: tensor<4x3x3x3xi8>, %stem__bias: tensor<4xi32>, %stem__m0: tensor<4xi32>, %dw__filter: tensor<3x3x4xi8>, %dw__bias: tensor<4xi32>, %dw__m0: tensor<4xi32>, %mid__filter: tensor<4x1x1x4xi8>, %mid__bias: tensor<4xi32>, %mid__m0: tensor<4xi32>, %head__filter: tensor<3x1x1x4xi8>, %head__bias: tensor<3xi32>, %head__m0: tensor<3xi32>) -> tensor<1x12x12x3xi8> {
    %zero_i32 = arith.constant 0 : i32
    %stem__padded = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0] {
      ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
        %c_h_pad = arith.constant 1 : index
        %c_h_zero = arith.constant 0 : index
        %c_h_n = arith.constant 6 : index
        %c_h_mirror = arith.constant 10 : index
        %h_shifted = arith.subi %i1, %c_h_pad : index
        %h_negative = arith.cmpi slt, %h_shifted, %c_h_zero : index
        %h_absolute = arith.subi %c_h_zero, %h_shifted : index
        %h_folded = arith.select %h_negative, %h_absolute, %h_shifted : index
        %h_overflow = arith.cmpi sge, %h_folded, %c_h_n : index
        %h_mirrored = arith.subi %c_h_mirror, %h_folded : index
        %h_source = arith.select %h_overflow, %h_mirrored, %h_folded : index
        %c_w_pad = arith.constant 1 : index
        %c_w_zero = arith.constant 0 : index
        %c_w_n = arith.constant 6 : index
        %c_w_mirror = arith.constant 10 : index
        %w_shifted = arith.subi %i2, %c_w_pad : index
        %w_negative = arith.cmpi slt, %w_shifted, %c_w_zero : index
        %w_absolute = arith.subi %c_w_zero, %w_shifted : index
        %w_folded = arith.select %w_negative, %w_absolute, %w_shifted : index
        %w_overflow = arith.cmpi sge, %w_folded, %c_w_n : index
        %w_mirrored = arith.subi %c_w_mirror, %w_folded : index
        %w_source = arith.select %w_overflow, %w_mirrored, %w_folded : index
        %value = tensor.extract %input[%i0, %h_source, %w_source, %i3] : tensor<1x6x6x3xi8>
        tensor.yield %value : i8
    } : tensor<1x6x6x3xi8> to tensor<1x8x8x3xi8> loc("tessera:pad#0")
    %stem__acc_init = tensor.empty() : tensor<1x6x6x4xi32>
    %stem__acc_zero = linalg.fill ins(%zero_i32 : i32) outs(%stem__acc_init : tensor<1x6x6x4xi32>) -> tensor<1x6x6x4xi32>
    %stem__acc = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%stem__padded, %stem__filter : tensor<1x8x8x3xi8>, tensor<4x3x3x3xi8>) outs(%stem__acc_zero : tensor<1x6x6x4xi32>) -> tensor<1x6x6x4xi32> loc("tessera:conv#0")
    %stem__out_init = tensor.empty() : tensor<1x6x6x4xi8>
    %stem__out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%stem__acc, %stem__bias, %stem__m0 : tensor<1x6x6x4xi32>, tensor<4xi32>, tensor<4xi32>) outs(%stem__out_init : tensor<1x6x6x4xi8>) {
      ^bb0(%acc: i32, %bias: i32, %m0: i32, %old: i8):
        %acc_b = arith.addi %acc, %bias : i32
        %acc64 = arith.extsi %acc_b : i32 to i64
        %m0_64 = arith.extsi %m0 : i32 to i64
        %product = arith.muli %acc64, %m0_64 : i64
        %zero64 = arith.constant 0 : i64
        %nonnegative = arith.cmpi sge, %product, %zero64 : i64
        %round_positive = arith.constant 2097152 : i64
        %round_negative = arith.constant 2097151 : i64
        %rounding = arith.select %nonnegative, %round_positive, %round_negative : i64
        %rounded = arith.addi %product, %rounding : i64
        %shift64 = arith.constant 22 : i64
        %scaled = arith.shrsi %rounded, %shift64 : i64
        %lower = arith.constant 0 : i64
        %saturated_low = arith.maxsi %scaled, %lower : i64
        %upper = arith.constant 127 : i64
        %saturated = arith.minsi %saturated_low, %upper : i64
        %narrow = arith.trunci %saturated : i64 to i8
        linalg.yield %narrow : i8
    } -> tensor<1x6x6x4xi8> loc("tessera:epilogue#0")
    %dw__padded = tensor.pad %stem__out low[0, 1, 1, 0] high[0, 1, 1, 0] {
      ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
        %c_h_pad = arith.constant 1 : index
        %c_h_zero = arith.constant 0 : index
        %c_h_n = arith.constant 6 : index
        %c_h_mirror = arith.constant 10 : index
        %h_shifted = arith.subi %i1, %c_h_pad : index
        %h_negative = arith.cmpi slt, %h_shifted, %c_h_zero : index
        %h_absolute = arith.subi %c_h_zero, %h_shifted : index
        %h_folded = arith.select %h_negative, %h_absolute, %h_shifted : index
        %h_overflow = arith.cmpi sge, %h_folded, %c_h_n : index
        %h_mirrored = arith.subi %c_h_mirror, %h_folded : index
        %h_source = arith.select %h_overflow, %h_mirrored, %h_folded : index
        %c_w_pad = arith.constant 1 : index
        %c_w_zero = arith.constant 0 : index
        %c_w_n = arith.constant 6 : index
        %c_w_mirror = arith.constant 10 : index
        %w_shifted = arith.subi %i2, %c_w_pad : index
        %w_negative = arith.cmpi slt, %w_shifted, %c_w_zero : index
        %w_absolute = arith.subi %c_w_zero, %w_shifted : index
        %w_folded = arith.select %w_negative, %w_absolute, %w_shifted : index
        %w_overflow = arith.cmpi sge, %w_folded, %c_w_n : index
        %w_mirrored = arith.subi %c_w_mirror, %w_folded : index
        %w_source = arith.select %w_overflow, %w_mirrored, %w_folded : index
        %value = tensor.extract %stem__out[%i0, %h_source, %w_source, %i3] : tensor<1x6x6x4xi8>
        tensor.yield %value : i8
    } : tensor<1x6x6x4xi8> to tensor<1x8x8x4xi8> loc("tessera:pad#2")
    %dw__acc_init = tensor.empty() : tensor<1x6x6x4xi32>
    %dw__acc_zero = linalg.fill ins(%zero_i32 : i32) outs(%dw__acc_init : tensor<1x6x6x4xi32>) -> tensor<1x6x6x4xi32>
    %dw__acc = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%dw__padded, %dw__filter : tensor<1x8x8x4xi8>, tensor<3x3x4xi8>) outs(%dw__acc_zero : tensor<1x6x6x4xi32>) -> tensor<1x6x6x4xi32> loc("tessera:conv#2")
    %dw__out_init = tensor.empty() : tensor<1x6x6x4xi8>
    %dw__out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%dw__acc, %dw__bias, %dw__m0 : tensor<1x6x6x4xi32>, tensor<4xi32>, tensor<4xi32>) outs(%dw__out_init : tensor<1x6x6x4xi8>) {
      ^bb0(%acc: i32, %bias: i32, %m0: i32, %old: i8):
        %acc_b = arith.addi %acc, %bias : i32
        %acc64 = arith.extsi %acc_b : i32 to i64
        %m0_64 = arith.extsi %m0 : i32 to i64
        %product = arith.muli %acc64, %m0_64 : i64
        %zero64 = arith.constant 0 : i64
        %nonnegative = arith.cmpi sge, %product, %zero64 : i64
        %round_positive = arith.constant 4194304 : i64
        %round_negative = arith.constant 4194303 : i64
        %rounding = arith.select %nonnegative, %round_positive, %round_negative : i64
        %rounded = arith.addi %product, %rounding : i64
        %shift64 = arith.constant 23 : i64
        %scaled = arith.shrsi %rounded, %shift64 : i64
        %lower = arith.constant 0 : i64
        %saturated_low = arith.maxsi %scaled, %lower : i64
        %upper = arith.constant 127 : i64
        %saturated = arith.minsi %saturated_low, %upper : i64
        %narrow = arith.trunci %saturated : i64 to i8
        linalg.yield %narrow : i8
    } -> tensor<1x6x6x4xi8> loc("tessera:epilogue#2")
    %mid__acc_init = tensor.empty() : tensor<1x6x6x4xi32>
    %mid__acc_zero = linalg.fill ins(%zero_i32 : i32) outs(%mid__acc_init : tensor<1x6x6x4xi32>) -> tensor<1x6x6x4xi32>
    %mid__acc = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%dw__out, %mid__filter : tensor<1x6x6x4xi8>, tensor<4x1x1x4xi8>) outs(%mid__acc_zero : tensor<1x6x6x4xi32>) -> tensor<1x6x6x4xi32> loc("tessera:conv#4")
    %mid__out_init = tensor.empty() : tensor<1x6x6x4xi8>
    %mid__out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mid__acc, %mid__bias, %mid__m0 : tensor<1x6x6x4xi32>, tensor<4xi32>, tensor<4xi32>) outs(%mid__out_init : tensor<1x6x6x4xi8>) {
      ^bb0(%acc: i32, %bias: i32, %m0: i32, %old: i8):
        %acc_b = arith.addi %acc, %bias : i32
        %acc64 = arith.extsi %acc_b : i32 to i64
        %m0_64 = arith.extsi %m0 : i32 to i64
        %product = arith.muli %acc64, %m0_64 : i64
        %zero64 = arith.constant 0 : i64
        %nonnegative = arith.cmpi sge, %product, %zero64 : i64
        %round_positive = arith.constant 4194304 : i64
        %round_negative = arith.constant 4194303 : i64
        %rounding = arith.select %nonnegative, %round_positive, %round_negative : i64
        %rounded = arith.addi %product, %rounding : i64
        %shift64 = arith.constant 23 : i64
        %scaled = arith.shrsi %rounded, %shift64 : i64
        %lower = arith.constant 0 : i64
        %saturated_low = arith.maxsi %scaled, %lower : i64
        %upper = arith.constant 127 : i64
        %saturated = arith.minsi %saturated_low, %upper : i64
        %narrow = arith.trunci %saturated : i64 to i8
        linalg.yield %narrow : i8
    } -> tensor<1x6x6x4xi8> loc("tessera:epilogue#4")
    %add_6__init = tensor.empty() : tensor<1x6x6x4xi16>
    %add_6__out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mid__out, %stem__out : tensor<1x6x6x4xi8>, tensor<1x6x6x4xi8>) outs(%add_6__init : tensor<1x6x6x4xi16>) {
      ^bb0(%trunk: i8, %branch: i8, %old: i16):
        %branch32 = arith.extsi %branch : i8 to i32
        %branch64 = arith.extsi %branch32 : i32 to i64
        %m0_64 = arith.constant 122334 : i64
        %product = arith.muli %branch64, %m0_64 : i64
        %zero64 = arith.constant 0 : i64
        %nonnegative = arith.cmpi sge, %product, %zero64 : i64
        %round_positive = arith.constant 131072 : i64
        %round_negative = arith.constant 131071 : i64
        %rounding = arith.select %nonnegative, %round_positive, %round_negative : i64
        %rounded = arith.addi %product, %rounding : i64
        %shift64 = arith.constant 18 : i64
        %scaled = arith.shrsi %rounded, %shift64 : i64
        %lower = arith.constant -127 : i64
        %saturated_low = arith.maxsi %scaled, %lower : i64
        %upper = arith.constant 127 : i64
        %saturated = arith.minsi %saturated_low, %upper : i64
        %narrow = arith.trunci %saturated : i64 to i8
        %trunk32 = arith.extsi %trunk : i8 to i32
        %narrow32 = arith.extsi %narrow : i8 to i32
        %sum32 = arith.addi %trunk32, %narrow32 : i32
        %sum = arith.trunci %sum32 : i32 to i16
        linalg.yield %sum : i16
    } -> tensor<1x6x6x4xi16> loc("tessera:add#6")
    %up_7__init = tensor.empty() : tensor<1x12x12x4xi16>
    %up_7__out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 2, d2 floordiv 2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%add_6__out : tensor<1x6x6x4xi16>) outs(%up_7__init : tensor<1x12x12x4xi16>) {
      ^bb0(%value: i16, %old: i16):
        linalg.yield %value : i16
    } -> tensor<1x12x12x4xi16> loc("tessera:upsample#7")
    %head__acc_init = tensor.empty() : tensor<1x12x12x3xi32>
    %head__acc_zero = linalg.fill ins(%zero_i32 : i32) outs(%head__acc_init : tensor<1x12x12x3xi32>) -> tensor<1x12x12x3xi32>
    %head__acc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 1 + d4, d2 * 1 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%up_7__out, %head__filter : tensor<1x12x12x4xi16>, tensor<3x1x1x4xi8>) outs(%head__acc_zero : tensor<1x12x12x3xi32>) {
      ^bb0(%activation: i16, %weight: i8, %accumulator: i32):
        %activation32 = arith.extsi %activation : i16 to i32
        %weight32 = arith.extsi %weight : i8 to i32
        %product = arith.muli %activation32, %weight32 : i32
        %sum = arith.addi %accumulator, %product : i32
        linalg.yield %sum : i32
    } -> tensor<1x12x12x3xi32> loc("tessera:conv#8")
    %head__out_init = tensor.empty() : tensor<1x12x12x3xi8>
    %head__out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%head__acc, %head__bias, %head__m0 : tensor<1x12x12x3xi32>, tensor<3xi32>, tensor<3xi32>) outs(%head__out_init : tensor<1x12x12x3xi8>) {
      ^bb0(%acc: i32, %bias: i32, %m0: i32, %old: i8):
        %acc_b = arith.addi %acc, %bias : i32
        %acc64 = arith.extsi %acc_b : i32 to i64
        %m0_64 = arith.extsi %m0 : i32 to i64
        %product = arith.muli %acc64, %m0_64 : i64
        %zero64 = arith.constant 0 : i64
        %nonnegative = arith.cmpi sge, %product, %zero64 : i64
        %round_positive = arith.constant 524288 : i64
        %round_negative = arith.constant 524287 : i64
        %rounding = arith.select %nonnegative, %round_positive, %round_negative : i64
        %rounded = arith.addi %product, %rounding : i64
        %shift64 = arith.constant 20 : i64
        %scaled = arith.shrsi %rounded, %shift64 : i64
        %lower = arith.constant 0 : i64
        %saturated_low = arith.maxsi %scaled, %lower : i64
        %upper = arith.constant 255 : i64
        %saturated = arith.minsi %saturated_low, %upper : i64
        %narrow = arith.trunci %saturated : i64 to i8
        linalg.yield %narrow : i8
    } -> tensor<1x12x12x3xi8> loc("tessera:epilogue#8")
    return %head__out : tensor<1x12x12x3xi8>
  }
}
