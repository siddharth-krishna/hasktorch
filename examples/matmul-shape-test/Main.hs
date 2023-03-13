{-# LANGUAGE DataKinds #-}

module Main where

import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Typed as T

main :: IO ()
main = do
  let x = T.ones :: T.CPUTensor 'D.Float '[2, 3]
  let y = T.ones :: T.CPUTensor 'D.Float '[5, 4]
  -- TODO this should be a compile error?
  print $ T.matmul x y
