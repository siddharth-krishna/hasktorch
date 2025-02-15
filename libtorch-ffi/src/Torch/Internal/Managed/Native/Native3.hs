
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Managed.Native.Native3 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Native.Native3 as Unmanaged


eye_ll
  :: Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
eye_ll = _cast2 Unmanaged.eye_ll

eye_out_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
eye_out_tl = _cast2 Unmanaged.eye_out_tl

eye_out_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
eye_out_tll = _cast3 Unmanaged.eye_out_tll

flatten_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
flatten_tll = _cast3 Unmanaged.flatten_tll

flatten_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
flatten_tl = _cast2 Unmanaged.flatten_tl

flatten_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
flatten_t = _cast1 Unmanaged.flatten_t

flatten_tlln
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
flatten_tlln = _cast4 Unmanaged.flatten_tlln

flatten_tnnn
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
flatten_tnnn = _cast4 Unmanaged.flatten_tnnn

flatten_tNn
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
flatten_tNn = _cast3 Unmanaged.flatten_tNn

fill__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
fill__ts = _cast2 Unmanaged.fill__ts

fill__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fill__tt = _cast2 Unmanaged.fill__tt

floor_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
floor_t = _cast1 Unmanaged.floor_t

floor__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
floor__t = _cast1 Unmanaged.floor__t

floor_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
floor_out_tt = _cast2 Unmanaged.floor_out_tt

floor_divide_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
floor_divide_tt = _cast2 Unmanaged.floor_divide_tt

floor_divide_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
floor_divide_out_ttt = _cast3 Unmanaged.floor_divide_out_ttt

floor_divide_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
floor_divide_ts = _cast2 Unmanaged.floor_divide_ts

frac_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
frac_t = _cast1 Unmanaged.frac_t

frac__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
frac__t = _cast1 Unmanaged.frac__t

frac_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
frac_out_tt = _cast2 Unmanaged.frac_out_tt

full_lsNo
  :: ForeignPtr IntArray
  -> ForeignPtr Scalar
  -> ForeignPtr DimnameList
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
full_lsNo = _cast4 Unmanaged.full_lsNo

full_lsN
  :: ForeignPtr IntArray
  -> ForeignPtr Scalar
  -> ForeignPtr DimnameList
  -> IO (ForeignPtr Tensor)
full_lsN = _cast3 Unmanaged.full_lsN

full_lso
  :: ForeignPtr IntArray
  -> ForeignPtr Scalar
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
full_lso = _cast3 Unmanaged.full_lso

full_ls
  :: ForeignPtr IntArray
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
full_ls = _cast2 Unmanaged.full_ls

full_out_tls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
full_out_tls = _cast3 Unmanaged.full_out_tls

full_like_tsoM
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr TensorOptions
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
full_like_tsoM = _cast4 Unmanaged.full_like_tsoM

full_like_tso
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
full_like_tso = _cast3 Unmanaged.full_like_tso

full_like_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
full_like_ts = _cast2 Unmanaged.full_like_ts

from_file_sblo
  :: ForeignPtr StdString
  -> CBool
  -> Int64
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
from_file_sblo = _cast4 Unmanaged.from_file_sblo

from_file_sbl
  :: ForeignPtr StdString
  -> CBool
  -> Int64
  -> IO (ForeignPtr Tensor)
from_file_sbl = _cast3 Unmanaged.from_file_sbl

from_file_sb
  :: ForeignPtr StdString
  -> CBool
  -> IO (ForeignPtr Tensor)
from_file_sb = _cast2 Unmanaged.from_file_sb

from_file_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
from_file_s = _cast1 Unmanaged.from_file_s

gcd_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
gcd_out_ttt = _cast3 Unmanaged.gcd_out_ttt

gcd_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
gcd_tt = _cast2 Unmanaged.gcd_tt

gcd__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
gcd__tt = _cast2 Unmanaged.gcd__tt

lcm_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
lcm_out_ttt = _cast3 Unmanaged.lcm_out_ttt

lcm_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
lcm_tt = _cast2 Unmanaged.lcm_tt

lcm__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
lcm__tt = _cast2 Unmanaged.lcm__tt

grid_sampler_ttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
grid_sampler_ttllb = _cast5 Unmanaged.grid_sampler_ttllb

grid_sampler_2d_ttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
grid_sampler_2d_ttllb = _cast5 Unmanaged.grid_sampler_2d_ttllb

grid_sampler_2d_backward_tttllba
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> ForeignPtr (StdArray '(CBool,2))
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
grid_sampler_2d_backward_tttllba = _cast7 Unmanaged.grid_sampler_2d_backward_tttllba

_grid_sampler_2d_cpu_fallback_ttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
_grid_sampler_2d_cpu_fallback_ttllb = _cast5 Unmanaged._grid_sampler_2d_cpu_fallback_ttllb

_grid_sampler_2d_cpu_fallback_backward_tttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
_grid_sampler_2d_cpu_fallback_backward_tttllb = _cast6 Unmanaged._grid_sampler_2d_cpu_fallback_backward_tttllb

grid_sampler_3d_ttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
grid_sampler_3d_ttllb = _cast5 Unmanaged.grid_sampler_3d_ttllb

grid_sampler_3d_backward_tttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
grid_sampler_3d_backward_tttllb = _cast6 Unmanaged.grid_sampler_3d_backward_tttllb

hann_window_lo
  :: Int64
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
hann_window_lo = _cast2 Unmanaged.hann_window_lo

hann_window_l
  :: Int64
  -> IO (ForeignPtr Tensor)
hann_window_l = _cast1 Unmanaged.hann_window_l

hann_window_lbo
  :: Int64
  -> CBool
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
hann_window_lbo = _cast3 Unmanaged.hann_window_lbo

hann_window_lb
  :: Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
hann_window_lb = _cast2 Unmanaged.hann_window_lb

hamming_window_lo
  :: Int64
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
hamming_window_lo = _cast2 Unmanaged.hamming_window_lo

hamming_window_l
  :: Int64
  -> IO (ForeignPtr Tensor)
hamming_window_l = _cast1 Unmanaged.hamming_window_l

hamming_window_lbo
  :: Int64
  -> CBool
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
hamming_window_lbo = _cast3 Unmanaged.hamming_window_lbo

hamming_window_lb
  :: Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
hamming_window_lb = _cast2 Unmanaged.hamming_window_lb

hamming_window_lbdo
  :: Int64
  -> CBool
  -> CDouble
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
hamming_window_lbdo = _cast4 Unmanaged.hamming_window_lbdo

hamming_window_lbd
  :: Int64
  -> CBool
  -> CDouble
  -> IO (ForeignPtr Tensor)
hamming_window_lbd = _cast3 Unmanaged.hamming_window_lbd

hamming_window_lbddo
  :: Int64
  -> CBool
  -> CDouble
  -> CDouble
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
hamming_window_lbddo = _cast5 Unmanaged.hamming_window_lbddo

hamming_window_lbdd
  :: Int64
  -> CBool
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
hamming_window_lbdd = _cast4 Unmanaged.hamming_window_lbdd

kaiser_window_lo
  :: Int64
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
kaiser_window_lo = _cast2 Unmanaged.kaiser_window_lo

kaiser_window_l
  :: Int64
  -> IO (ForeignPtr Tensor)
kaiser_window_l = _cast1 Unmanaged.kaiser_window_l

kaiser_window_lbo
  :: Int64
  -> CBool
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
kaiser_window_lbo = _cast3 Unmanaged.kaiser_window_lbo

kaiser_window_lb
  :: Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
kaiser_window_lb = _cast2 Unmanaged.kaiser_window_lb

kaiser_window_lbdo
  :: Int64
  -> CBool
  -> CDouble
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
kaiser_window_lbdo = _cast4 Unmanaged.kaiser_window_lbdo

kaiser_window_lbd
  :: Int64
  -> CBool
  -> CDouble
  -> IO (ForeignPtr Tensor)
kaiser_window_lbd = _cast3 Unmanaged.kaiser_window_lbd

hinge_embedding_loss_ttdl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> Int64
  -> IO (ForeignPtr Tensor)
hinge_embedding_loss_ttdl = _cast4 Unmanaged.hinge_embedding_loss_ttdl

hinge_embedding_loss_ttd
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
hinge_embedding_loss_ttd = _cast3 Unmanaged.hinge_embedding_loss_ttd

hinge_embedding_loss_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
hinge_embedding_loss_tt = _cast2 Unmanaged.hinge_embedding_loss_tt

group_norm_tlttdb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CBool
  -> IO (ForeignPtr Tensor)
group_norm_tlttdb = _cast6 Unmanaged.group_norm_tlttdb

group_norm_tlttd
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
group_norm_tlttd = _cast5 Unmanaged.group_norm_tlttd

group_norm_tltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
group_norm_tltt = _cast4 Unmanaged.group_norm_tltt

group_norm_tlt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
group_norm_tlt = _cast3 Unmanaged.group_norm_tlt

group_norm_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
group_norm_tl = _cast2 Unmanaged.group_norm_tl

native_group_norm_tttlllld
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> CDouble
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
native_group_norm_tttlllld = _cast8 Unmanaged.native_group_norm_tttlllld

native_group_norm_backward_tttttlllla
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> ForeignPtr (StdArray '(CBool,3))
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
native_group_norm_backward_tttttlllla = _cast10 Unmanaged.native_group_norm_backward_tttttlllla

_fft_r2c_tllb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
_fft_r2c_tllb = _cast4 Unmanaged._fft_r2c_tllb

_fft_r2c_out_ttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
_fft_r2c_out_ttllb = _cast5 Unmanaged._fft_r2c_out_ttllb

_fft_c2r_tlll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
_fft_c2r_tlll = _cast4 Unmanaged._fft_c2r_tlll

_fft_c2r_out_ttlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
_fft_c2r_out_ttlll = _cast5 Unmanaged._fft_c2r_out_ttlll

_fft_c2c_tllb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
_fft_c2c_tllb = _cast4 Unmanaged._fft_c2c_tllb

_fft_c2c_out_ttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
_fft_c2c_out_ttllb = _cast5 Unmanaged._fft_c2c_out_ttllb

_cufft_get_plan_cache_size_l
  :: Int64
  -> IO (Int64)
_cufft_get_plan_cache_size_l = _cast1 Unmanaged._cufft_get_plan_cache_size_l

_cufft_get_plan_cache_max_size_l
  :: Int64
  -> IO (Int64)
_cufft_get_plan_cache_max_size_l = _cast1 Unmanaged._cufft_get_plan_cache_max_size_l

_cufft_set_plan_cache_max_size_ll
  :: Int64
  -> Int64
  -> IO (())
_cufft_set_plan_cache_max_size_ll = _cast2 Unmanaged._cufft_set_plan_cache_max_size_ll

_cufft_clear_plan_cache_l
  :: Int64
  -> IO (())
_cufft_clear_plan_cache_l = _cast1 Unmanaged._cufft_clear_plan_cache_l

index_tl
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> IO (ForeignPtr Tensor)
index_tl = _cast2 Unmanaged.index_tl

index_copy_tltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
index_copy_tltt = _cast4 Unmanaged.index_copy_tltt

index_copy_tntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
index_copy_tntt = _cast4 Unmanaged.index_copy_tntt

index_put__tltb
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
index_put__tltb = _cast4 Unmanaged.index_put__tltb

index_put__tlt
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
index_put__tlt = _cast3 Unmanaged.index_put__tlt

index_put_tltb
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
index_put_tltb = _cast4 Unmanaged.index_put_tltb

index_put_tlt
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
index_put_tlt = _cast3 Unmanaged.index_put_tlt

_index_put_impl__tltbb
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
_index_put_impl__tltbb = _cast5 Unmanaged._index_put_impl__tltbb

_index_put_impl__tltb
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
_index_put_impl__tltb = _cast4 Unmanaged._index_put_impl__tltb

_index_put_impl__tlt
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
_index_put_impl__tlt = _cast3 Unmanaged._index_put_impl__tlt

instance_norm_tttttbddb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (ForeignPtr Tensor)
instance_norm_tttttbddb = _cast9 Unmanaged.instance_norm_tttttbddb

inverse_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
inverse_t = _cast1 Unmanaged.inverse_t

inverse_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
inverse_out_tt = _cast2 Unmanaged.inverse_out_tt

isclose_ttddb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (ForeignPtr Tensor)
isclose_ttddb = _cast5 Unmanaged.isclose_ttddb

isclose_ttdd
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
isclose_ttdd = _cast4 Unmanaged.isclose_ttdd

isclose_ttd
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
isclose_ttd = _cast3 Unmanaged.isclose_ttd

isclose_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
isclose_tt = _cast2 Unmanaged.isclose_tt

isin_out_tttbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_out_tttbb = _cast5 Unmanaged.isin_out_tttbb

isin_out_tttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_out_tttb = _cast4 Unmanaged.isin_out_tttb

isin_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
isin_out_ttt = _cast3 Unmanaged.isin_out_ttt

isin_ttbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_ttbb = _cast4 Unmanaged.isin_ttbb

isin_ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_ttb = _cast3 Unmanaged.isin_ttb

isin_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
isin_tt = _cast2 Unmanaged.isin_tt

isin_out_ttsbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_out_ttsbb = _cast5 Unmanaged.isin_out_ttsbb

isin_out_ttsb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_out_ttsb = _cast4 Unmanaged.isin_out_ttsb

isin_out_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
isin_out_tts = _cast3 Unmanaged.isin_out_tts

isin_tsbb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_tsbb = _cast4 Unmanaged.isin_tsbb

isin_tsb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_tsb = _cast3 Unmanaged.isin_tsb

isin_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
isin_ts = _cast2 Unmanaged.isin_ts

isin_out_tstbb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_out_tstbb = _cast5 Unmanaged.isin_out_tstbb

isin_out_tstb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_out_tstb = _cast4 Unmanaged.isin_out_tstb

isin_out_tst
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
isin_out_tst = _cast3 Unmanaged.isin_out_tst

isin_stbb
  :: ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_stbb = _cast4 Unmanaged.isin_stbb

isin_stb
  :: ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
isin_stb = _cast3 Unmanaged.isin_stb

isin_st
  :: ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
isin_st = _cast2 Unmanaged.isin_st

isnan_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
isnan_t = _cast1 Unmanaged.isnan_t

is_distributed_t
  :: ForeignPtr Tensor
  -> IO (CBool)
is_distributed_t = _cast1 Unmanaged.is_distributed_t

is_floating_point_t
  :: ForeignPtr Tensor
  -> IO (CBool)
is_floating_point_t = _cast1 Unmanaged.is_floating_point_t

is_complex_t
  :: ForeignPtr Tensor
  -> IO (CBool)
is_complex_t = _cast1 Unmanaged.is_complex_t

is_conj_t
  :: ForeignPtr Tensor
  -> IO (CBool)
is_conj_t = _cast1 Unmanaged.is_conj_t

_is_zerotensor_t
  :: ForeignPtr Tensor
  -> IO (CBool)
_is_zerotensor_t = _cast1 Unmanaged._is_zerotensor_t

is_neg_t
  :: ForeignPtr Tensor
  -> IO (CBool)
is_neg_t = _cast1 Unmanaged.is_neg_t

isreal_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
isreal_t = _cast1 Unmanaged.isreal_t

is_nonzero_t
  :: ForeignPtr Tensor
  -> IO (CBool)
is_nonzero_t = _cast1 Unmanaged.is_nonzero_t

is_same_size_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (CBool)
is_same_size_tt = _cast2 Unmanaged.is_same_size_tt

is_signed_t
  :: ForeignPtr Tensor
  -> IO (CBool)
is_signed_t = _cast1 Unmanaged.is_signed_t

is_inference_t
  :: ForeignPtr Tensor
  -> IO (CBool)
is_inference_t = _cast1 Unmanaged.is_inference_t

kl_div_ttlb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
kl_div_ttlb = _cast4 Unmanaged.kl_div_ttlb

kl_div_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
kl_div_ttl = _cast3 Unmanaged.kl_div_ttl

kl_div_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
kl_div_tt = _cast2 Unmanaged.kl_div_tt

kl_div_backward_tttlb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
kl_div_backward_tttlb = _cast5 Unmanaged.kl_div_backward_tttlb

kl_div_backward_tttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
kl_div_backward_tttl = _cast4 Unmanaged.kl_div_backward_tttl

kl_div_backward_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
kl_div_backward_ttt = _cast3 Unmanaged.kl_div_backward_ttt

kron_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
kron_tt = _cast2 Unmanaged.kron_tt

kron_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
kron_out_ttt = _cast3 Unmanaged.kron_out_ttt

kthvalue_tllb
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_tllb = _cast4 Unmanaged.kthvalue_tllb

kthvalue_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_tll = _cast3 Unmanaged.kthvalue_tll

kthvalue_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_tl = _cast2 Unmanaged.kthvalue_tl

kthvalue_out_tttllb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_out_tttllb = _cast6 Unmanaged.kthvalue_out_tttllb

kthvalue_out_tttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_out_tttll = _cast5 Unmanaged.kthvalue_out_tttll

kthvalue_out_tttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_out_tttl = _cast4 Unmanaged.kthvalue_out_tttl

kthvalue_tlnb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_tlnb = _cast4 Unmanaged.kthvalue_tlnb

kthvalue_tln
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Dimname
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_tln = _cast3 Unmanaged.kthvalue_tln

kthvalue_out_tttlnb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_out_tttlnb = _cast6 Unmanaged.kthvalue_out_tttlnb

kthvalue_out_tttln
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Dimname
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
kthvalue_out_tttln = _cast5 Unmanaged.kthvalue_out_tttln

layer_norm_tlttdb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CBool
  -> IO (ForeignPtr Tensor)
layer_norm_tlttdb = _cast6 Unmanaged.layer_norm_tlttdb

layer_norm_tlttd
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
layer_norm_tlttd = _cast5 Unmanaged.layer_norm_tlttd

layer_norm_tltt
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
layer_norm_tltt = _cast4 Unmanaged.layer_norm_tltt

layer_norm_tlt
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
layer_norm_tlt = _cast3 Unmanaged.layer_norm_tlt

layer_norm_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
layer_norm_tl = _cast2 Unmanaged.layer_norm_tl

native_layer_norm_tlttd
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
native_layer_norm_tlttd = _cast5 Unmanaged.native_layer_norm_tlttd

_native_multi_head_self_attention_tttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
_native_multi_head_self_attention_tttttt = _cast6 Unmanaged._native_multi_head_self_attention_tttttt

_native_multi_head_self_attention_ttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
_native_multi_head_self_attention_ttttt = _cast5 Unmanaged._native_multi_head_self_attention_ttttt

native_layer_norm_backward_ttltttta
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr (StdArray '(CBool,3))
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
native_layer_norm_backward_ttltttta = _cast8 Unmanaged.native_layer_norm_backward_ttltttta

nan_to_num_tddd
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num_tddd = _cast4 Unmanaged.nan_to_num_tddd

nan_to_num_tdd
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num_tdd = _cast3 Unmanaged.nan_to_num_tdd

nan_to_num_td
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num_td = _cast2 Unmanaged.nan_to_num_td

nan_to_num_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
nan_to_num_t = _cast1 Unmanaged.nan_to_num_t

nan_to_num__tddd
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num__tddd = _cast4 Unmanaged.nan_to_num__tddd

nan_to_num__tdd
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num__tdd = _cast3 Unmanaged.nan_to_num__tdd

nan_to_num__td
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num__td = _cast2 Unmanaged.nan_to_num__td

nan_to_num__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
nan_to_num__t = _cast1 Unmanaged.nan_to_num__t

nan_to_num_out_ttddd
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num_out_ttddd = _cast5 Unmanaged.nan_to_num_out_ttddd

nan_to_num_out_ttdd
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num_out_ttdd = _cast4 Unmanaged.nan_to_num_out_ttdd

nan_to_num_out_ttd
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
nan_to_num_out_ttd = _cast3 Unmanaged.nan_to_num_out_ttd

nan_to_num_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
nan_to_num_out_tt = _cast2 Unmanaged.nan_to_num_out_tt

linear_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linear_ttt = _cast3 Unmanaged.linear_ttt

linear_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linear_tt = _cast2 Unmanaged.linear_tt

linear_out_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linear_out_tttt = _cast4 Unmanaged.linear_out_tttt

linear_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linear_out_ttt = _cast3 Unmanaged.linear_out_ttt

mkldnn_linear_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
mkldnn_linear_ttt = _cast3 Unmanaged.mkldnn_linear_ttt

mkldnn_linear_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
mkldnn_linear_tt = _cast2 Unmanaged.mkldnn_linear_tt

mkldnn_linear_backward_input_ltt
  :: ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
mkldnn_linear_backward_input_ltt = _cast3 Unmanaged.mkldnn_linear_backward_input_ltt

mkldnn_linear_backward_weights_tttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
mkldnn_linear_backward_weights_tttb = _cast4 Unmanaged.mkldnn_linear_backward_weights_tttb

mkldnn_linear_backward_ttta
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr (StdArray '(CBool,3))
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
mkldnn_linear_backward_ttta = _cast4 Unmanaged.mkldnn_linear_backward_ttta

fbgemm_linear_int8_weight_fp32_activation_ttttsst
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fbgemm_linear_int8_weight_fp32_activation_ttttsst = _cast7 Unmanaged.fbgemm_linear_int8_weight_fp32_activation_ttttsst

fbgemm_linear_int8_weight_ttttsst
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fbgemm_linear_int8_weight_ttttsst = _cast7 Unmanaged.fbgemm_linear_int8_weight_ttttsst

fbgemm_linear_quantize_weight_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,CDouble,Int64)))
fbgemm_linear_quantize_weight_t = _cast1 Unmanaged.fbgemm_linear_quantize_weight_t

fbgemm_pack_gemm_matrix_fp16_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fbgemm_pack_gemm_matrix_fp16_t = _cast1 Unmanaged.fbgemm_pack_gemm_matrix_fp16_t

fbgemm_linear_fp16_weight_fp32_activation_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fbgemm_linear_fp16_weight_fp32_activation_ttt = _cast3 Unmanaged.fbgemm_linear_fp16_weight_fp32_activation_ttt

fbgemm_linear_fp16_weight_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fbgemm_linear_fp16_weight_ttt = _cast3 Unmanaged.fbgemm_linear_fp16_weight_ttt

fbgemm_pack_quantized_matrix_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fbgemm_pack_quantized_matrix_t = _cast1 Unmanaged.fbgemm_pack_quantized_matrix_t

fbgemm_pack_quantized_matrix_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fbgemm_pack_quantized_matrix_tll = _cast3 Unmanaged.fbgemm_pack_quantized_matrix_tll

ldexp_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
ldexp_tt = _cast2 Unmanaged.ldexp_tt

ldexp__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
ldexp__tt = _cast2 Unmanaged.ldexp__tt

ldexp_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
ldexp_out_ttt = _cast3 Unmanaged.ldexp_out_ttt

linspace_sslo
  :: ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> Int64
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
linspace_sslo = _cast4 Unmanaged.linspace_sslo

linspace_ssl
  :: ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> Int64
  -> IO (ForeignPtr Tensor)
linspace_ssl = _cast3 Unmanaged.linspace_ssl

linspace_out_tssl
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> Int64
  -> IO (ForeignPtr Tensor)
linspace_out_tssl = _cast4 Unmanaged.linspace_out_tssl

log_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log_t = _cast1 Unmanaged.log_t

log__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log__t = _cast1 Unmanaged.log__t

log_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log_out_tt = _cast2 Unmanaged.log_out_tt

log10_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log10_t = _cast1 Unmanaged.log10_t

log10__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log10__t = _cast1 Unmanaged.log10__t

log10_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log10_out_tt = _cast2 Unmanaged.log10_out_tt

log1p_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log1p_t = _cast1 Unmanaged.log1p_t

log1p__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log1p__t = _cast1 Unmanaged.log1p__t

log1p_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log1p_out_tt = _cast2 Unmanaged.log1p_out_tt

log2_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log2_t = _cast1 Unmanaged.log2_t

log2__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log2__t = _cast1 Unmanaged.log2__t

log2_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
log2_out_tt = _cast2 Unmanaged.log2_out_tt

logaddexp_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logaddexp_out_ttt = _cast3 Unmanaged.logaddexp_out_ttt

logaddexp_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logaddexp_tt = _cast2 Unmanaged.logaddexp_tt

logaddexp2_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logaddexp2_out_ttt = _cast3 Unmanaged.logaddexp2_out_ttt

logaddexp2_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logaddexp2_tt = _cast2 Unmanaged.logaddexp2_tt

xlogy_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
xlogy_tt = _cast2 Unmanaged.xlogy_tt

xlogy_st
  :: ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
xlogy_st = _cast2 Unmanaged.xlogy_st

xlogy_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
xlogy_ts = _cast2 Unmanaged.xlogy_ts

xlogy__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
xlogy__tt = _cast2 Unmanaged.xlogy__tt

xlogy__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
xlogy__ts = _cast2 Unmanaged.xlogy__ts

xlogy_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
xlogy_out_ttt = _cast3 Unmanaged.xlogy_out_ttt

