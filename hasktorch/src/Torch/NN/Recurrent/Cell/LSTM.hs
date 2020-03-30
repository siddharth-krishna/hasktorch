{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.LSTM where

import Control.Monad.State.Strict
import GHC.Generics

import Torch
import Torch.NN.Recurrent.Cell.RecurrentLayer

data LSTMSpec = LSTMSpec { inf :: Int, hf :: Int}

data LSTMCell = LSTMCell {
	weightIH :: Parameter,
	weightHH :: Parameter,
	biasIH :: Parameter,
	biasHH :: Parameter
} deriving (Generic, Show)

lstmCellForward :: LSTMCell -> Tensor -> (Tensor, Tensor) -> (Tensor, Tensor)
lstmCellForward LSTMCell{..} input hidden =
	lstmCell weightIH' weightHH' biasIH' biasHH' hidden input
	where
		weightIH' = toDependent weightIH
		weightHH' = toDependent weightHH
		biasIH' = toDependent biasIH
		biasHH' = toDependent biasHH

instance Randomizable LSTMSpec LSTMCell where
  sample LSTMSpec{..} = do
    weightIH' <- makeIndependent =<< randIO' [inf, hf]
    weightHH' <- makeIndependent =<< randIO' [hf, hf]
    biasIH' <- makeIndependent =<< randIO' [hf]
    biasHH' <- makeIndependent =<< randIO' [hf]
    pure $ LSTMCell {
        weightIH=weightIH',
        weightHH=weightHH',
        biasIH=biasIH',
        biasHH=biasHH' }
