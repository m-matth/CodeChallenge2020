{-# LANGUAGE ScopedTypeVariables  #-}
module Main where

import Lib
import Tensor
import Environment
import Control.Monad.IO.Class (MonadIO, liftIO)

main :: IO ()
main = do
  (env :: GameEnvironment) <- basicEnv
  results <- runEnv env action boardX boardY
  putStrLn "Done!"
  where
    -- action = resetEnv >> (gameRenderLoop chooseActionUser :: FrozenLake (FrozenLakeObservation, Reward))
    action = (resetEnv >> gameLearningIterations :: ChallengeGame [Reward])
