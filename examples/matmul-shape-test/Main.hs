{-# LANGUAGE DataKinds #-}

module Main where

import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Typed as T

main :: IO ()
main = do
  let x = T.ones :: T.CPUTensor 'D.Float '[2, 3]
  let y = T.ones :: T.CPUTensor 'D.Float '[5, 4]
  -- TODO this should be a compile error, but isn't
  print $ T.matmul x y
  -- This line gives us the expected compile error
  let z = T.matmul x y :: T.CPUTensor 'D.Float '[2, 4]
  print $ z
