{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GMultiHeadAttention where

import Control.Monad.Catch (MonadThrow)
import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Control.Monad.State (evalStateT)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import qualified Data.Map.Strict as Map
import Data.Singletons (SingKind (..))
import Data.Singletons.Prelude.List (SList (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.NN.Linear (GLinear (..), LinearBiasF, LinearWeightF, linearSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (SWithBias, SWithoutBias))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, sGetDimFromShape, sUnifyDim, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SBy (..), SDim (..), SName (..), SSelectDim (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, sReshape, sTranspose)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF, matmul)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, mulScalar)
import Torch.GraduallyTyped.Tensor.Type (SGetShape (..), Tensor (..), TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Data type for representing whether or not (and, if so, where) scaling is applied in the multi-headed attention layer.
data MultiHeadAttentionHasScaling
  = -- | Scaling is not done.
    MultiHeadAttentionWithoutScaling
  | -- | Scaling is applied to the query after in the in-projection.
    MultiHeadAttentionWithQueryScaling
  | -- | Scaling is applied to the attention weights.
    MultiHeadAttentionWithWeightScaling
  deriving stock (Eq, Ord, Show, Generic)

-- | Generic multi-headed attention layer.
--
-- - @headDim@ is the dimension of the attention heads.
-- - @headEmbedDim@ is the dimension of the attention head embedding.
-- - @embedDim@ is the dimension of the embedding.
-- - @qInProj@ is the type of the query projection.
-- - @kInProj@ is the type of the key projection.
-- - @vInProj@ is the type of the value projection.
-- - @outProj@ is the type of the output projection.
-- - @dropout@ is the type of the dropout layer.
data
  GMultiHeadAttention
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (qInProj :: Type)
    (kInProj :: Type)
    (vInProj :: Type)
    (outProj :: Type)
    (dropout :: Type)
  where
  GMultiHeadAttention ::
    forall headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout.
    { -- | head dim
      mhaHeadDim :: SDim headDim,
      -- | head embed dim
      mhaHeadEmbedDim :: SDim headEmbedDim,
      -- | embed dim
      mhaEmbedDim :: SDim embedDim,
      -- | in-projection for query
      mhaQInProj :: qInProj,
      -- | in-projection for key
      mhaKInProj :: kInProj,
      -- | in-projection for value
      mhaVInProj :: vInProj,
      -- | out-projection
      mhaOutProj :: outProj,
      -- | dropout
      mhaDropout :: dropout,
      -- | scaling
      mhaScaling :: MultiHeadAttentionHasScaling
    } ->
    GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout
  deriving stock (Show)

type instance
  ModelSpec (GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout) =
    GMultiHeadAttention headDim headEmbedDim embedDim (ModelSpec qInProj) (ModelSpec kInProj) (ModelSpec vInProj) (ModelSpec outProj) (ModelSpec dropout)

-- | Specifies the linear transformation of the query.
type family
  QInProjF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  QInProjF 'T5 gradient device dataType queryEmbedDim embedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType queryEmbedDim embedDim))
          (NamedModel (LinearBiasF 'WithoutBias gradient device dataType embedDim))
      )
  QInProjF 'ByT5 gradient device dataType queryEmbedDim embedDim =
    QInProjF 'T5 gradient device dataType queryEmbedDim embedDim
  QInProjF _ gradient device dataType queryEmbedDim embedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType queryEmbedDim embedDim))
          (NamedModel (LinearBiasF 'WithBias gradient device dataType embedDim))
      )

-- | Specifies the linear transformation of the key.
type family
  KInProjF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  KInProjF 'T5 gradient device dataType keyEmbedDim embedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType keyEmbedDim embedDim))
          (NamedModel (LinearBiasF 'WithoutBias gradient device dataType embedDim))
      )
  KInProjF 'ByT5 gradient device dataType keyEmbedDim embedDim =
    KInProjF 'T5 gradient device dataType keyEmbedDim embedDim
  KInProjF _ gradient device dataType keyEmbedDim embedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType keyEmbedDim embedDim))
          (NamedModel (LinearBiasF 'WithBias gradient device dataType embedDim))
      )

-- | Specifies the linear transformation of the value.
type family
  VInProjF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  VInProjF 'T5 gradient device dataType valueEmbedDim embedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType valueEmbedDim embedDim))
          (NamedModel (LinearBiasF 'WithoutBias gradient device dataType embedDim))
      )
  VInProjF 'ByT5 gradient device dataType valueEmbedDim embedDim =
    VInProjF 'T5 gradient device dataType valueEmbedDim embedDim
  VInProjF _ gradient device dataType valueEmbedDim embedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType valueEmbedDim embedDim))
          (NamedModel (LinearBiasF 'WithBias gradient device dataType embedDim))
      )

-- | Specifies the type of the out-projection layer.
type family
  OutProjF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  OutProjF 'T5 gradient device dataType embedDim queryEmbedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType embedDim queryEmbedDim))
          (NamedModel (LinearBiasF 'WithoutBias gradient device dataType queryEmbedDim))
      )
  OutProjF 'ByT5 gradient device dataType embedDim queryEmbedDim =
    OutProjF 'T5 gradient device dataType embedDim queryEmbedDim
  OutProjF _ gradient device dataType embedDim queryEmbedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType embedDim queryEmbedDim))
          (NamedModel (LinearBiasF 'WithBias gradient device dataType queryEmbedDim))
      )

-- | Specifies the type of the dropout layer.
type family
  DropoutF
    (style :: TransformerStyle) ::
    Type
  where
  DropoutF _ = Dropout

-- | Specifies the parameters of a multi-headed attention layer.
--
-- - @style@: the style of the attention layer, e.g. 'ST5', 'ByT5', etc.
-- - @gradient@: whether to compute the gradient of the attention layer.
-- - @device@: the computational device on which to allocate the attention layer.
-- - @dataType@: the data type of the attention layer.
-- - @headDim@: the dimension of the attention heads.
-- - @headEmbedDim@: the dimension of the attention head embeddings.
-- - @embedDim@: the dimension of the input embeddings.
-- - @queryEmbedDim@: the dimension of the query embeddings.
-- - @keyEmbedDim@: the dimension of the key embeddings.
-- - @valueEmbedDim@: the dimension of the value embeddings.
-- - @dropoutP@: the dropout rate.
multiHeadAttentionSpec ::
  forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim queryEmbedDim ->
  SDim keyEmbedDim ->
  SDim valueEmbedDim ->
  Double ->
  ModelSpec
    ( GMultiHeadAttention
        headDim
        headEmbedDim
        embedDim
        (QInProjF style gradient device dataType queryEmbedDim embedDim)
        (KInProjF style gradient device dataType keyEmbedDim embedDim)
        (VInProjF style gradient device dataType valueEmbedDim embedDim)
        (OutProjF style gradient device dataType embedDim queryEmbedDim)
        (DropoutF style)
    )
multiHeadAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =
  let qInProjSpec ST5 = NamedModel "q." (projSpecWithoutBias queryEmbedDim embedDim)
      qInProjSpec SByT5 = NamedModel "q." (projSpecWithoutBias queryEmbedDim embedDim)
      qInProjSpec SBART = NamedModel "q_proj." (projSpecWithBias queryEmbedDim embedDim)
      qInProjSpec SMBART = NamedModel "q_proj." (projSpecWithBias queryEmbedDim embedDim)
      qInProjSpec SPegasus = NamedModel "q_proj." (projSpecWithBias queryEmbedDim embedDim)
      qInProjSpec SBERT = NamedModel "self.query." (projSpecWithBias queryEmbedDim embedDim)
      qInProjSpec SRoBERTa = NamedModel "self.query." (projSpecWithBias queryEmbedDim embedDim)
      qInProjSpec SGPT2 = undefined
      kInProjSpec ST5 = NamedModel "k." (projSpecWithoutBias keyEmbedDim embedDim)
      kInProjSpec SByT5 = NamedModel "k." (projSpecWithoutBias keyEmbedDim embedDim)
      kInProjSpec SBART = NamedModel "k_proj." (projSpecWithBias keyEmbedDim embedDim)
      kInProjSpec SMBART = NamedModel "k_proj." (projSpecWithBias keyEmbedDim embedDim)
      kInProjSpec SPegasus = NamedModel "k_proj." (projSpecWithBias keyEmbedDim embedDim)
      kInProjSpec SBERT = NamedModel "self.key." (projSpecWithBias keyEmbedDim embedDim)
      kInProjSpec SRoBERTa = NamedModel "self.key." (projSpecWithBias keyEmbedDim embedDim)
      kInProjSpec SGPT2 = undefined
      vInProjSpec ST5 = NamedModel "v." (projSpecWithoutBias valueEmbedDim embedDim)
      vInProjSpec SByT5 = NamedModel "v." (projSpecWithoutBias valueEmbedDim embedDim)
      vInProjSpec SBART = NamedModel "v_proj." (projSpecWithBias valueEmbedDim embedDim)
      vInProjSpec SMBART = NamedModel "v_proj." (projSpecWithBias valueEmbedDim embedDim)
      vInProjSpec SPegasus = NamedModel "v_proj." (projSpecWithBias valueEmbedDim embedDim)
      vInProjSpec SBERT = NamedModel "self.value." (projSpecWithBias valueEmbedDim embedDim)
      vInProjSpec SRoBERTa = NamedModel "self.value." (projSpecWithBias valueEmbedDim embedDim)
      vInProjSpec SGPT2 = undefined
      outProjSpec ST5 = NamedModel "o." (projSpecWithoutBias embedDim queryEmbedDim)
      outProjSpec SByT5 = NamedModel "o." (projSpecWithoutBias embedDim queryEmbedDim)
      outProjSpec SBART = NamedModel "out_proj." (projSpecWithBias embedDim queryEmbedDim)
      outProjSpec SMBART = NamedModel "out_proj." (projSpecWithBias embedDim queryEmbedDim)
      outProjSpec SPegasus = NamedModel "out_proj." (projSpecWithBias embedDim queryEmbedDim)
      outProjSpec SBERT = NamedModel "output.dense." (projSpecWithBias embedDim queryEmbedDim)
      outProjSpec SRoBERTa = NamedModel "output.dense." (projSpecWithBias embedDim queryEmbedDim)
      outProjSpec SGPT2 = undefined
      dropoutSpec _ = Dropout dropoutP
      scaling :: STransformerStyle style -> MultiHeadAttentionHasScaling
      scaling ST5 = MultiHeadAttentionWithoutScaling
      scaling SByT5 = MultiHeadAttentionWithoutScaling
      scaling SBART = MultiHeadAttentionWithQueryScaling
      scaling SMBART = MultiHeadAttentionWithQueryScaling
      scaling SPegasus = MultiHeadAttentionWithQueryScaling
      scaling SBERT = MultiHeadAttentionWithWeightScaling
      scaling SRoBERTa = MultiHeadAttentionWithWeightScaling
      scaling SGPT2 = undefined
   in GMultiHeadAttention
        headDim
        headEmbedDim
        embedDim
        (qInProjSpec style)
        (kInProjSpec style)
        (vInProjSpec style)
        (outProjSpec style)
        (dropoutSpec style)
        (scaling style)
  where
    projSpecWithoutBias ::
      forall inputDim outputDim.
      SDim inputDim ->
      SDim outputDim ->
      ModelSpec
        ( GLinear
            (NamedModel (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])))
            (NamedModel ())
        )
    projSpecWithoutBias = linearSpec SWithoutBias gradient device dataType
    projSpecWithBias ::
      forall inputDim outputDim.
      SDim inputDim ->
      SDim outputDim ->
      ModelSpec
        ( GLinear
            (NamedModel (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])))
            (NamedModel (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim])))
        )
    projSpecWithBias = linearSpec SWithBias gradient device dataType

instance
  ( HasInitialize qInProj generatorDevice qInProj generatorDevice,
    HasInitialize kInProj generatorDevice kInProj generatorDevice,
    HasInitialize vInProj generatorDevice vInProj generatorDevice,
    HasInitialize outProj generatorDevice outProj generatorDevice,
    HasInitialize dropout generatorDevice dropout generatorDevice
  ) =>
  HasInitialize
    (GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout)
    generatorDevice
    (GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout)
    generatorDevice
  where
  initialize (GMultiHeadAttention headDim headEmbedDim embedDim qInProjSpec kInProjSpec vInProjSpec outProjSpec dropoutSpec scaling) =
    let qInProj = IxStateT . initialize $ qInProjSpec
        kInProj = IxStateT . initialize $ kInProjSpec
        vInProj = IxStateT . initialize $ vInProjSpec
        outProj = IxStateT . initialize $ outProjSpec
        dropout = IxStateT . initialize $ dropoutSpec
     in runIxStateT $
          GMultiHeadAttention
            <<$>> ireturn headDim
            <<*>> ireturn headEmbedDim
            <<*>> ireturn embedDim
            <<*>> qInProj
            <<*>> kInProj
            <<*>> vInProj
            <<*>> outProj
            <<*>> dropout
            <<*>> ireturn scaling

instance
  ( HasStateDict qInProj,
    HasStateDict vInProj,
    HasStateDict kInProj,
    HasStateDict outProj,
    HasStateDict dropout
  ) =>
  HasStateDict
    (GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout)
  where
  fromStateDict (GMultiHeadAttention headDim headEmbedDim embedDim qInProjSpec kInProjSpec vInProjSpec outProjSpec dropoutSpec scaling) k =
    GMultiHeadAttention headDim headEmbedDim embedDim
      <$> fromStateDict qInProjSpec k
      <*> fromStateDict kInProjSpec k
      <*> fromStateDict vInProjSpec k
      <*> fromStateDict outProjSpec k
      <*> fromStateDict dropoutSpec k
      <*> pure scaling
  toStateDict k GMultiHeadAttention {..} = do
    () <- toStateDict k mhaQInProj
    () <- toStateDict k mhaKInProj
    () <- toStateDict k mhaVInProj
    () <- toStateDict k mhaOutProj
    () <- toStateDict k mhaDropout
    pure ()

type BatchDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type BatchDim queryShape keyShape valueShape =
  (queryShape ! 0) <+> (keyShape ! 0) <+> (valueShape ! 0)

getBatchDim ::
  forall m queryShape keyShape valueShape batchDim.
  (MonadThrow m, batchDim ~ BatchDim queryShape keyShape valueShape) =>
  SShape queryShape ->
  SShape keyShape ->
  SShape valueShape ->
  m (SDim batchDim)
getBatchDim queryShape keyShape valueShape = do
  queryBatchDim <- sGetDimFromShape (SSelectDim $ SByIndex @0) queryShape
  keyBatchDim <- sGetDimFromShape (SSelectDim $ SByIndex @0) keyShape
  valueBatchDim <- sGetDimFromShape (SSelectDim $ SByIndex @0) valueShape
  keyValueBatchDim <- sUnifyDim keyBatchDim valueBatchDim
  sUnifyDim queryBatchDim keyValueBatchDim

type QuerySeqDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type QuerySeqDim queryShape =
  queryShape ! 1

getQuerySeqDim ::
  forall m queryShape querySeqDim.
  (MonadThrow m, querySeqDim ~ QuerySeqDim queryShape) =>
  SShape queryShape ->
  m (SDim querySeqDim)
getQuerySeqDim = sGetDimFromShape (SSelectDim $ SByIndex @1)

type KeySeqDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type KeySeqDim keyShape valueShape =
  (keyShape ! 1) <+> (valueShape ! 1)

getKeySeqDim ::
  forall m keyShape valueShape keySeqDim.
  (MonadThrow m, keySeqDim ~ KeySeqDim keyShape valueShape) =>
  SShape keyShape ->
  SShape valueShape ->
  m (SDim keySeqDim)
getKeySeqDim keyShape valueShape =
  do
    keySeqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) keyShape
    valueSeqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) valueShape
    sUnifyDim keySeqDim valueSeqDim

-- | 'HasForward' instance for 'GMultiHeadAttention'.
--
-- @
-- ┌───────────────┐        ┌───────┐       ┌─────┐       ┌───────┐
-- │ attentionBias │        │ query │       │ key │       │ value │
-- └───────┬───────┘        └───┬───┘       └──┬──┘       └───┬───┘
--         │                    │              │              │
--         │                    ▼              ▼              ▼
--         │                mhaQInProj     mhaKInProj     mhaVInProj
--         │                    ▼              │              │
--         │                (scaling)          │              │
--         │                    ▼              ▼              ▼
--         │                 reshape        reshape        reshape
--         │                    ▼              ▼              ▼
--         │                transpose      transpose      transpose
--         │                    │              ▼              │
--         │                    │          transpose          │
--         │                    │              │              │
--         │                    └───►matmul◄───┘              │
--         │                           ▼                      │
--         │                       (scaling)                  │
--         │                           │                      │
--         └──────────►add◄────────────┘                      │
--                      ▼                                     │
--                   softmax                                  │
--                      ▼                                     │
--                  mhaDropout                                │
--                      │                                     │
--                      └──────────────►matmul◄───────────────┘
--                                        ▼
--                                    transpose
--                                        ▼
--                                     reshape
--                                        ▼
--                                    mhaOutProj
--                                        │
--                                        ▼
--                                    ┌───────┐
--                                    │ query │
--                                    └───────┘
-- @
instance
  ( HasForward
      qInProj
      (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
      generatorDevice
      (Tensor qRequiresGradient qLayout qDevice qDataType qShape0)
      qGeneratorOutputDevice,
    qShape
      ~ TransposeF
          ('SelectDim ('ByIndex 1))
          ('SelectDim ('ByIndex 2))
          ( ReshapeF
              qShape0
              ('Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
          ),
    HasForward
      kInProj
      (Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape)
      qGeneratorOutputDevice
      (Tensor qRequiresGradient kLayout kDevice kDataType kShape0)
      kGeneratorOutputDevice,
    weightsShape0
      ~ SoftmaxF
          ('SelectDim ('ByIndex 3))
          ( BroadcastShapesF
              ( MatmulF
                  qShape
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      ( TransposeF
                          ('SelectDim ('ByIndex 1))
                          ('SelectDim ('ByIndex 2))
                          ( ReshapeF
                              kShape0
                              ('Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
                          )
                      )
                  )
              )
              attentionBiasShape
          ),
    HasForward
      dropout
      ( Tensor
          (qRequiresGradient <|> attentionBiasRequiresGradient)
          (qLayout <+> kLayout <+> attentionBiasLayout)
          (qDevice <+> kDevice <+> attentionBiasDevice)
          (qDataType <+> kDataType <+> attentionBiasDataType)
          weightsShape0
      )
      kGeneratorOutputDevice
      (Tensor weightsRequiresGradient weightsLayout weightsDevice weightsDataType weightsShape)
      weightsGeneratorOutputDevice,
    HasForward
      vInProj
      (Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape)
      weightsGeneratorOutputDevice
      (Tensor weightsRequiresGradient vLayout vDevice vDataType vShape0)
      vGeneratorOutputDevice,
    outputQueryShape0
      ~ TransposeF
          ('SelectDim ('ByIndex 1))
          ('SelectDim ('ByIndex 2))
          ( MatmulF
              weightsShape
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( ReshapeF
                      vShape0
                      ('Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
                  )
              )
          ),
    HasForward
      outProj
      ( Tensor
          weightsRequiresGradient
          (weightsLayout <+> vLayout)
          (weightsDevice <+> vDevice)
          (weightsDataType <+> vDataType)
          (ReshapeF outputQueryShape0 ('Shape '[batchDim, querySeqDim, embedDim]))
      )
      vGeneratorOutputDevice
      output
      generatorOutputDevice,
    SGetShape queryShape,
    SGetShape keyShape,
    SGetShape valueShape,
    batchDim ~ BatchDim queryShape keyShape valueShape,
    querySeqDim ~ QuerySeqDim queryShape,
    keySeqDim ~ KeySeqDim keyShape valueShape
  ) =>
  HasForward
    (GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GMultiHeadAttention {..} (query, key, value, attentionBias) g = do
    batchDim <-
      let queryShape = sGetShape query
          keyShape = sGetShape key
          valueShape = sGetShape value
       in getBatchDim queryShape keyShape valueShape
    querySeqDim <-
      let queryShape = sGetShape query
       in getQuerySeqDim queryShape
    keySeqDim <-
      let keyShape = sGetShape key
          valueShape = sGetShape value
       in getKeySeqDim keyShape valueShape
    let scaling = (1 :: Double) / (sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ mhaHeadEmbedDim)
    flip runIxStateT g $
      let q =
            ireturn query
              >>>= IxStateT . forward mhaQInProj
              >>>= ireturn
                . ( \case
                      MultiHeadAttentionWithoutScaling -> id
                      MultiHeadAttentionWithQueryScaling -> flip mulScalar scaling
                      MultiHeadAttentionWithWeightScaling -> id
                  )
                  mhaScaling
              >>>= ireturn . sReshape (SShape $ batchDim :|: querySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ilift . sTranspose (SSelectDim (SByIndex @1)) (SSelectDim (SByIndex @2))
          k =
            ireturn key
              >>>= IxStateT . forward mhaKInProj
              >>>= ireturn . sReshape (SShape $ batchDim :|: keySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ilift . sTranspose (SSelectDim (SByIndex @1)) (SSelectDim (SByIndex @2))
          kt = k >>>= ilift . sTranspose (SSelectDim (SByIndex @2)) (SSelectDim (SByIndex @3))
          weights =
            matmul <<$>> q <<*>> kt
              >>>= ireturn
                . ( \case
                      MultiHeadAttentionWithoutScaling -> id
                      MultiHeadAttentionWithQueryScaling -> id
                      MultiHeadAttentionWithWeightScaling -> flip mulScalar scaling
                  )
                  mhaScaling
              >>>= ireturn . (`add` attentionBias)
              >>>= IxStateT . forward mhaDropout . softmax (SSelectDim (SByIndex @3))
          v =
            ireturn value
              >>>= IxStateT . forward mhaVInProj
              >>>= ireturn . sReshape (SShape $ batchDim :|: keySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ilift . sTranspose (SSelectDim (SByIndex @1)) (SSelectDim (SByIndex @2))
       in matmul <<$>> weights <<*>> v
            >>>= ilift . sTranspose (SSelectDim (SByIndex @1)) (SSelectDim (SByIndex @2))
            >>>= ireturn . sReshape (SShape $ batchDim :|: querySeqDim :|: mhaEmbedDim :|: SNil)
            >>>= IxStateT . forward mhaOutProj

testMHA :: IO _
testMHA = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @2
      headEmbedDim = SName @"*" :&: SSize @2
      embedDim = SName @"*" :&: SSize @4
      queryEmbedDim = SName @"*" :&: SSize @3
      keyEmbedDim = SName @"*" :&: SSize @5
      valueEmbedDim = SName @"*" :&: SSize @7
      dropoutP = 0
  let g = sMkGenerator device 0
      spec = NamedModel "mha." $ multiHeadAttentionSpec SByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP
  (mha, g') <- initialize spec g
  mha' <- flip evalStateT Map.empty $ do
    toStateDict mempty mha
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @2
      seqDim = SName @"*" :&: SSize @1
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      value = sOnes' dataType (SShape $ batchDim :|: seqDim :|: valueEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward mha' (query, key, value, attentionBias) g'
  pure output
