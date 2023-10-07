{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (when)
import Data.Bitraversable (bimapM)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Typed as T

--------------------------------------------------------------------------------
-- Basics
--------------------------------------------------------------------------------

{-

Hasktorch is a package providing Haskell bindings to libtorch,
which is a C++ library used by PyTorch to do tensor computations and ML stuff
at speed / on GPUs

Tensor: a multidimensional array with fixed shape and element type
Examples:
  zeros' ([3, 4] :: [Int])
  asTensor ([[4, 3], [2, 1]] :: [[Float]])

Scalar values are represented in Hasktorch as tensors with shape []:
asTensor (3.5 :: Float)

We can get the scalar value back out using asValue:
asValue (asTensor (3.5 :: Float)) :: Float

DType and Device can be specified on unprimed constructors:
zeros [4, 4] (withDType Int64 defaultOpts)
zeros [4, 4] (withDevice (CUDA 0) defaultOpts)

Basic operations on tensors:
let x = ones' [4] in x + x

let x = ones' [2, 3] in let y = ones' [3, 4] in matmul x y

relu (asTensor ([-1.0, -0.5, 0.5, 1] :: [Float]))

-}

--------------------------------------------------------------------------------
-- Automatic Differentiation
--------------------------------------------------------------------------------

{-

Hasktorch piggybacks on Torch's AD.

You mark certain tensors as "independent",
(think variables with respect to which the gradient is computed)
these will usually be the parameters of your ML model,
and then just perform tensor operations on these.

makeIndependent :: Tensor -> IO IndependentTensor

newtype IndependentTensor = IndependentTensor {toDependent :: Tensor} deriving (Show)

x <- makeIndependent (asTensor ([5 :: Float]))
let x' = toDependent x
let y = asTensor ([1 :: Float])
let z = x' * x' + y
requiresGrad z
requiresGrad y
-- z = x^2 + y so dz/dx = 2x = 10
grad z [x]
grad y [x]

Behind the scenes, a computation graph is built up so that `grad` can backprop
and compute gradients efficiently.

-}

--------------------------------------------------------------------------------
-- DIY Linear Regression
--------------------------------------------------------------------------------

-- y = m * x + b

diyRegression :: IO ()
diyRegression = do
  -- init = (m, b)
  init <- bimapM makeIndependent makeIndependent (zeros' [1], zeros' [1])
  randGen <- defaultRNG
  (trained, _) <- foldLoop (init, randGen) numIters $ \(state, randGen) i -> do
    let (input, randGen') = randn' [batchSize] randGen
        (y, y') = (groundTruthFn input, modelFn state input)
        loss = sumAll $ pow (2 :: Int) (y - y')
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (newParam, _) <- runStep state optimizer loss 5e-3
    pure (newParam, randGen')
  putStrLn $ show $ trained
  pure ()
  where
    groundTruthFn x = 1.1 * x + 0.3 -- TODO + random noise
    modelFn (m, b) x = (toDependent m) * x + (toDependent b)
    optimizer = GD
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 4
    numIters = 2000

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }

data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)

{-
class Parameterized f where
  flattenParameters :: f -> [Parameter]
  default flattenParameters :: (Generic f, Parameterized' (Rep f)) => f -> [Parameter]
  flattenParameters f = flattenParameters' (from f)

  replaceOwnParameters :: f -> ParamStream f
  default replaceOwnParameters :: (Generic f, Parameterized' (Rep f)) => f -> ParamStream f
  replaceOwnParameters f = to <$> replaceOwnParameters' (from f)
-}

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

batchSize = 2

numIters = 2000

model :: MLP -> Tensor -> Tensor
model params t = mlp params t

mlpMain :: IO ()
mlpMain = do
  initModel <-
    sample $
      MLPSpec
        { feature_counts = [2, 2, 1],
          nonlinearitySpec = Torch.tanh
        }
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)

  (trainedModel, _) <- foldLoop (initModel, initOptim) numIters $ \(state, optim) i -> do
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
    let (y, y') = (tensorXOR input, squeezeAll $ model state input)
        loss = mseLoss y y'
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss

    (newState, newOptim) <- runStep state optim loss 1e-1
    return (newState, newOptim)

  putStrLn "Final Model:"
  putStrLn $ show $ layers trainedModel
  putStrLn $ "0, 0 => " ++ (show $ squeezeAll $ model trainedModel (asTensor [0, 0 :: Float]))
  putStrLn $ "0, 1 => " ++ (show $ squeezeAll $ model trainedModel (asTensor [0, 1 :: Float]))
  putStrLn $ "1, 0 => " ++ (show $ squeezeAll $ model trainedModel (asTensor [1, 0 :: Float]))
  putStrLn $ "1, 1 => " ++ (show $ squeezeAll $ model trainedModel (asTensor [1, 1 :: Float]))
  return ()
  where
    tensorXOR :: Tensor -> Tensor
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t

--------------------------------------------------------------------------------
-- Load & Serve
--------------------------------------------------------------------------------

-- See examples/model-serving/04-python-torchscript/

--------------------------------------------------------------------------------
-- Shape checking
--------------------------------------------------------------------------------

-- See examples/static-xor-mlp/

staticMain :: IO ()
staticMain = do
  let x = T.ones :: T.CPUTensor 'D.Float '[2, 3]
  let y = T.ones :: T.CPUTensor 'D.Float '[5]
  print $ T.matmul x y

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

gpuEx :: IO ()
gpuEx = do
  let n = 256
  let x = ones [n, n] (withDevice (Device CUDA 0) defaultOpts)
  let y = ones [n, n] (withDevice (Device CUDA 0) defaultOpts)
  print $ matmul x y

main :: IO ()
main = do
  -- diyRegression
  -- mlpMain
  staticMain
