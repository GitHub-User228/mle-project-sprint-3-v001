from pathlib import Path

import numpy as np
import pandas as pd
from typing import *
import featuretools as ft
from typing import Literal
from scipy.spatial import cKDTree
from sklearn.preprocessing import *
from featuretools.primitives import *
from sklearn.pipeline import Pipeline
from autofeat import AutoFeatRegressor
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector

from scripts import logger
from scripts.metrics import (
    spherical_to_cartesian,
    haversine_dist2,
)


class DistanceTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer that calculates the distance between a given point
    and the rows in a DataFrame.

    Attributes:
        point (List[float | int]):
            The reference point to calculate the distance from.
        in_features_names (List[str | int]):
            The names of the input features to use for the distance
            calculation.
        out_feature_name (str | int):
            The name of the output feature that will contain the
            calculated distances.
        distance_metric_name (str):
            The name of the distance metric function to use for the
            calculation.
        _point (numpy.ndarray):
            The reference point as a numpy array.
        _distance_metric (Callable):
            The distance metric function to use for the calculation.

    Raises:
        TypeError:
            If any of the input arguments are of the wrong type.
        ValueError:
            If any of the input arguments are invalid or inconsistent.
        Exception:
            If there is an error fitting or applying the transformer.

    """

    def __init__(
        self,
        point: List[float | int],
        in_features_names: List[str | int],
        out_feature_name: str | int,
        distance_metric_name: str,
    ) -> None:
        """
        Initializes a DistanceTransformer instance with the given
        parameters.

        Args:
            point (List[float | int]):
                The reference point to calculate the distance from.
            in_features_names (List[str | int]):
                The names of the input features to use for the distance
                calculation.
            out_feature_name (str | int):
                The name of the output feature that will contain the
                calculated distances.
            distance_metric_name (str):
                The name of the distance metric function to use for the
                calculation.
        """
        self.point = point
        self.in_features_names = in_features_names
        self.out_feature_name = out_feature_name
        self.distance_metric_name = distance_metric_name
        self._validate_input()
        self._point = np.array(point, dtype=np.float32).reshape(1, -1)
        self._distance_metric = globals()[distance_metric_name]

    def _validate_input(self) -> None:
        """
        Validates the input arguments.

        Raises:
            TypeError:
                If any of the input arguments are of the wrong type.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """
        if not isinstance(self.point, list):
            raise TypeError("point must be a list of floats or ints")
        if not all(isinstance(x, (int, float)) for x in self.point):
            raise TypeError("point must be a list of floats or ints")
        if not isinstance(self.in_features_names, list):
            raise TypeError("in_features_names must be a list of strings")
        if not all(isinstance(x, (str, int)) for x in self.in_features_names):
            raise TypeError(
                "in_features_names must be a list of strings or ints"
            )
        if not isinstance(self.out_feature_name, (str, int)):
            raise TypeError("out_feature_name must be a string or int")
        if not isinstance(self.distance_metric_name, str):
            raise TypeError("distance_metric_name must be a string")
        if len(self.in_features_names) != len(self.point):
            raise ValueError(
                "in_features_names and point must have the same length"
            )
        if self.distance_metric_name not in globals():
            raise ValueError(
                f"Distance metric {self.distance_metric_name} not found"
            )

    def fit(self, X: pd.DataFrame, y=None) -> "DistanceTransformer":
        """
        Fits the DistanceTransformer to the input DataFrame.

        Args:
            X (pd.DataFrame):
                The input DataFrame to fit the transformer on.
            y (optional):
                Ignored, as this transformer does not use the target
                variable.

        Returns:
            self:
                The fitted transformer.

        Raises:
            Exception:
                If some of the features specified in `in_features_names`
                are not present in the input DataFrame.
        """
        if not all(x in X.columns for x in self.in_features_names):
            raise Exception(
                f"Failed to fit {self.__class__.__name__}. Some features "
                f"from in_features_names are not in the input DataFrame"
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applies the specified distance metric to the input DataFrame
        and returns the distances as a new column.

        Args:
            X (pd.DataFrame):
                The input DataFrame

        Returns:
            np.ndarray:
                An array of calculated distances.

        Raises:
            Exception:
                If there is an error applying the distance metric to
                the input data.
        """
        try:
            return self._distance_metric(
                X[self.in_features_names].values, self._point
            ).reshape(-1, 1)
        except Exception as e:
            raise Exception(
                f"Failed to apply {self.__class__.__name__}"
            ) from e

    def get_feature_names_out(self, input_features=None) -> List[str | int]:
        """
        Returns the resulting feature name wrapped in a list.

        Returns:
            List[str | int]:
                A list containing the single output feature name.
        """
        return [self.out_feature_name]


class ClosestDistanceTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer that calculates the closest distance between each row in the
    input DataFrame and a set of predefined points.

    Args:
        points_csv_path (str):
            The path to a CSV file containing the predefined points.
        points_features_names (List[str | int]):
            The names of the features in the points CSV
            file that should be used to calculate the distances.
        in_features_names (List[str | int]):
            The names of the features in the input DataFrame that
            should be used to calculate the distances.
        out_feature_name (str):
            The name of the output feature that will contain the
            calculated distances.
        distance_metric_name (str):
            The name of the distance metric function to use.
        points_prep_func_name (str | None):
            The name of an optional preprocessing function to apply to
            the points before calculating the distances.
        _distance_metric (Callable):
            The distance metric function to use.
        _points_prep_func (Callable | None):
            The preprocessing function to apply to the points before
            calculating the distances.
        _points (pd.DataFrame):
            The predefined points.
        _tree (cKDTree):
            The cKDTree used to calculate the distances.


    Raises:
        TypeError:
            If any of the input arguments are of the wrong type.
        ValueError:
            If any of the input arguments are invalid or inconsistent.
        FileNotFoundError:
            If the points_csv_path does not exist.
        Exception:
            If there is an error fitting or applying the transformer.
    """

    def __init__(
        self,
        points_csv_path: str,
        points_features_names: List[str | int],
        in_features_names: List[str | int],
        out_feature_name: str,
        distance_metric_name: str,
        points_prep_func_name: str | None = None,
    ) -> None:
        """
        Initializes a ClosestDistanceTransformer instance.

        Args:
            points_csv_path (str):
                The path to a CSV file containing the predefined points.
            points_features_names (List[str | int]):
                The names of the features in the points CSV file that
                should be used to calculate the distances.
            in_features_names (List[str | int]):
                The names of the features in the input DataFrame that
                should be used to calculate the distances.
            out_feature_name (str):
                The name of the output feature that will contain the
                calculated distances.
            distance_metric_name (str):
                The name of the distance metric function to use.
            points_prep_func_name (str | None):
                The name of an optional preprocessing function to apply
                to the points before calculating the distances.
        """
        self.points_csv_path = points_csv_path
        self.points_features_names = points_features_names
        self.in_features_names = in_features_names
        self.out_feature_name = out_feature_name
        self.distance_metric_name = distance_metric_name
        self.points_prep_func_name = points_prep_func_name
        self._validate_input()
        self._distance_metric = globals()[distance_metric_name]
        if self.points_prep_func_name:
            self._points_preprocessing_func = globals()[points_prep_func_name]

    def _validate_input(
        self,
    ) -> None:
        """
        Validates the input arguments.

        Raises:
            TypeError:
                If any of the input arguments are of the wrong type.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
            FileNotFoundError:
                If the points_csv_path does not exist.
        """
        if not isinstance(self.points_csv_path, str):
            raise TypeError("points_csv_path must be a str")
        if not isinstance(self.points_features_names, list):
            raise TypeError("points_features_names must be a list of strings")
        if not isinstance(self.in_features_names, list):
            raise TypeError("in_features_names must be a list of strings")
        if not isinstance(self.out_feature_name, str):
            raise TypeError("out_feature_name must be a string")
        if not isinstance(self.distance_metric_name, str):
            raise TypeError("distance_metric_name must be a string")
        if self.distance_metric_name not in globals():
            raise ValueError(
                f"Distance metric {self.distance_metric_name} not found"
            )
        if (
            not isinstance(self.points_prep_func_name, str)
            and self.points_prep_func_name is not None
        ):
            raise TypeError("points_prep_func_name must be a str or None")
        if self.points_prep_func_name is not None:
            if self.points_prep_func_name not in globals():
                raise ValueError(
                    f"points_prep_func_name {self.points_prep_func_name} not found"
                )
        if not Path(self.points_csv_path).exists():
            raise FileNotFoundError(
                f"points_csv_path {self.points_csv_path} does not exist"
            )
        if not all(
            isinstance(x, (str, int)) for x in self.points_features_names
        ):
            raise TypeError("points_features_names must be a list of strings")
        if not all(isinstance(x, (str, int)) for x in self.in_features_names):
            raise TypeError("in_features_names must be a list of strings")
        if len(set(self.points_features_names)) != len(
            self.points_features_names
        ):
            raise ValueError("points_features_names must be unique")
        if len(set(self.in_features_names)) != len(self.in_features_names):
            raise ValueError("in_features_names must be unique")
        if len(
            set(self.in_features_names) - set(self.points_features_names)
        ) != len(self.in_features_names):
            raise ValueError(
                "in_features_names and points_features_names must be different"
            )
        if len(self.in_features_names) != len(self.points_features_names):
            raise ValueError(
                "in_features_names and points_features_names must have the same length"
            )

    def fit(self, X: pd.DataFrame, y=None) -> "ClosestDistanceTransformer":
        """
        Fits the transformer by loading the points data from the
        specified CSV file and building a cKD-tree for efficient
        nearest neighbor search.

        Args:
            X (pd.DataFrame):
                The input dataframe.
            y (optional):
                Ignored argument.

        Returns:
            self:
                The fitted transformer.

        Raises:
            Exception:
                If there is any error during the fitting process.
        """
        try:
            self._points = pd.read_csv(
                self.points_csv_path, usecols=self.points_features_names
            ).astype(np.float32)
            if self.points_prep_func_name:
                self._tree = cKDTree(
                    self._points_preprocessing_func(self._points.values)
                )
            else:
                self._tree = cKDTree(self._points.values)
            return self
        except Exception as e:
            raise Exception(f"Failed to fit {self.__class__.__name__}") from e

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input DataFrame `X` by calculating the distance
        to the nearest neighbor for each row in a set of pre-defined
        points.

        Args:
            X (pd.DataFrame):
                The input DataFrame.

        Returns:
            np.ndarray:
                An array of calculated distances.

        Raises:
            Exception:
                If there is any error during the transformation.
        """
        try:
            X2 = X[self.in_features_names].copy()
            if self.points_prep_func_name:
                X2["point_id"] = self._tree.query(
                    self._points_preprocessing_func(X2.values)
                )[1]
            else:
                X2["point_id"] = self._tree.query(X2.values)[1]
            X2 = X2.merge(
                self._points,
                left_on="point_id",
                right_index=True,
                how="left",
            )
            return self._distance_metric(
                X2[self.in_features_names].values,
                X2[self.points_features_names].values,
            ).reshape(-1, 1)
        except Exception as e:
            raise Exception(
                f"Failed to apply {self.__class__.__name__}"
            ) from e

    def get_feature_names_out(self, input_features=None) -> List[str | int]:
        """
        Returns the resulting feature name wrapped in a list.

        Returns:
            List[str | int]:
                A list containing the single output feature name.
        """
        return [self.out_feature_name]


class FeatureToolsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that uses FeatureTools to generate features from a
    DataFrame. Only transformation primitives are supported.

    Attributes:
        index_col (str | int):
            The name or index of the column to use as the index for
            the DataFrame.
        trans_primitives (List[str | Dict[str, Any]]):
            A list of transformation primitives to use. Each primitive
            can be either a string (name of the primitive) or a
            dict with the name of the primitive and its parameters.
        drop_contains (List[str] | None):
            A list of strings to use as a filter for columns to drop.
        primitive_options (Dict[str, Any] | None):
            A dictionary of options to pass to the primitive.
        n_jobs (int):
            The number of jobs to run in parallel.

    Raises:
        TypeError:
            If the input arguments are not of the expected type.
        ValueError:
            If the input arguments are not valid.
        Exception:
             If there is an error fitting or applying the transformer.
    """

    def __init__(
        self,
        trans_primitives: List[str | Dict[str, Any]],
        drop_contains: List[str] | None = None,
        primitive_options: Dict[str, Any] | None = None,
        n_jobs: int = 1,
    ) -> None:
        """
        Initializes a FeatureToolsTransformer instance.

        Args:
            tmp_index_col (str | int):
                The name or index of the column to use as the index
                for the DataFrame.
            trans_primitives (List[str | Dict[str, Any]]):
                A list of transformation primitives to use. Each
                primitive can be either a string (name of the primitive)
                or a dict with the name of the primitive and its
                parameters.
            drop_contains (List[str] | None, optional):
                A list of strings to use as a filter for columns to
                drop.
            primitive_options (Dict[str, Any] | None, optional):
                A dictionary of options to pass to the primitive.
            n_jobs (int, optional):
                The number of jobs to run in parallel.
        """
        self.trans_primitives = trans_primitives
        self.drop_contains = drop_contains
        self.primitive_options = primitive_options
        self.n_jobs = n_jobs
        self._validate_input()
        self._index_col = "__index_col__"

    def _validate_input(self) -> None:
        """
        Validates the input arguments.

        Raises:
            TypeError:
                If any of the input arguments are of the wrong type.
            ValueError:
                If any of the input arguments are invalid.
        """
        if not isinstance(self.trans_primitives, list):
            raise TypeError("trans_primitives must be a list")
        if not isinstance(self.drop_contains, (list, type(None))):
            raise TypeError("drop_contains must be a list or None")
        if self.drop_contains:
            if not all(isinstance(i, str) for i in self.drop_contains):
                raise TypeError("drop_contains must be a list of strings")
        if not isinstance(self.primitive_options, (dict, type(None))):
            raise TypeError("primitive_options must be a dict or None")
        if not isinstance(self.n_jobs, int):
            raise TypeError("n_jobs must be an integer")

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureToolsTransformer":
        """
        Fits the FeatureToolsTransformer instance by creating an
        EntitySet, adding the input DataFrame, and computing the
        feature names based on the specified transformations.

        Args:
            X (pd.DataFrame):
                The input DataFrame to fit the transformer on.
            y (optional):
                Ignored. Included for compatibility.

        Returns:
            self:
                The fitted transformer instance.

        Raises:
            Exception:
                If there is an error fitting the transformer.
        """
        try:
            n_input_cols = X.shape[1]
            self._trans_primitives = [
                (
                    globals()[k["name"]](**k["params"])
                    if isinstance(k, dict)
                    else k
                )
                for k in self.trans_primitives
            ]
            es = ft.EntitySet(id="data")
            es.add_dataframe(
                dataframe_name="main",
                dataframe=X.iloc[: min(10, len(X))],
                index=self._index_col,
                make_index=True,
            )
            self._feature_names = ft.dfs(
                entityset=es,
                target_dataframe_name="main",
                trans_primitives=self._trans_primitives,
                drop_contains=self.drop_contains,
                primitive_options=self.primitive_options,
                n_jobs=self.n_jobs,
            )[0].columns.tolist()[n_input_cols:]
            return self
        except Exception as e:
            raise Exception(f"Failed to fit {self.__class__.__name__}") from e

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input DataFrame by applying the specified
        feature transformations.

        Args:
            X (pd.DataFrame):
                The input DataFrame to transform.

        Returns:
            np.ndarray:
                The transformed feature values as a NumPy array.

        Raises:
            Exception:
                If there is an error applying the transformations.
        """
        try:
            n_input_cols = X.shape[1]
            es = ft.EntitySet(id="data")
            es.add_dataframe(
                dataframe_name="main",
                dataframe=X,
                index=self._index_col,
                make_index=True,
            )
            X_transformed = ft.dfs(
                entityset=es,
                target_dataframe_name="main",
                trans_primitives=self._trans_primitives,
                drop_contains=self.drop_contains,
                primitive_options=self.primitive_options,
                n_jobs=self.n_jobs,
            )[0][self._feature_names].values.astype(np.float32)
            logger.info(
                f"[{self.__class__.__name__}] [transform] Generated "
                f"{len(self._feature_names)} features out of {n_input_cols}."
            )
            return X_transformed
        except Exception as e:
            raise Exception(
                f"Failed to apply {self.__class__.__name__}"
            ) from e

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns the feature names of the transformed data.

        Returns:
            List[str]:
                The names of the transformed features.
        """
        return self._feature_names


class AutoFeatTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies automated feature engineering and
    selection to the input data.

    Attributes:
        feateng_steps (int):
            The number of feature engineering steps to perform.
        featsel_runs (int):
            The number of feature selection runs to perform.
        max_gb (int):
            The maximum amount of memory (in GB) to use for feature
            engineering.
        transformations (List[str]):
            The list of feature transformation functions to use.
        n_jobs (int):
            The number of parallel jobs to use for feature engineering
            and selection.
        verbose (bool):
            Whether to print verbose output.
        _afr (AutoFeatRegressor):
            The AutoFeatRegressor instance used for feature engineering
            and selection.

    Raises:
        TypeError:
            If any of the input arguments are of the wrong type.
        ValueError:
            If any of the input arguments are invalid.
        Exception:
             If there is an error fitting or applying the transformer.
    """

    def __init__(
        self,
        feateng_steps: int,
        featsel_runs: int,
        max_gb: int,
        transformations: List[str],
        corr_threshold: float | None = 0.9,
        keep_feateng_cols: bool = False,
        pass_feateng_cols: bool = False,
        n_jobs: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Initializes an instance of the `AutoFeatTransformer` class.

        Args:
            feateng_steps (int):
                The number of feature engineering steps to perform.
            featsel_runs (int):
                The number of feature selection runs to perform.
            max_gb (int):
                The maximum amount of memory (in GB) to use for feature
                engineering.
            transformations (List[str]):
                The list of feature transformation functions to use.
            n_jobs (int):
                The number of parallel jobs to use for feature engineering
                and selection.
            verbose (bool):
                Whether to print verbose output.
        """
        self.feateng_steps = feateng_steps
        self.featsel_runs = featsel_runs
        self.max_gb = max_gb
        self.transformations = transformations
        self.corr_threshold = corr_threshold
        self.keep_feateng_cols = keep_feateng_cols
        self.pass_feateng_cols = pass_feateng_cols
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._validate_input()
        self._afr = AutoFeatRegressor(
            feateng_steps=self.feateng_steps,
            featsel_runs=self.featsel_runs,
            max_gb=self.max_gb,
            transformations=self.transformations,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

    def _validate_input(self) -> None:
        if not isinstance(self.feateng_steps, int):
            raise TypeError("feateng_steps must be an integer.")
        if not isinstance(self.featsel_runs, int):
            raise TypeError("featsel_runs must be an integer.")
        if not isinstance(self.max_gb, int):
            raise TypeError("max_gb must be an integer.")
        if not isinstance(self.transformations, list):
            raise TypeError("transformations must be a list of strings.")
        if not all(isinstance(t, str) for t in self.transformations):
            raise TypeError("transformations must be a list of strings.")
        if not isinstance(self.n_jobs, int):
            raise TypeError("n_jobs must be an integer.")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean.")
        if isinstance(self.feateng_steps, int) and self.feateng_steps <= 0:
            raise ValueError("feateng_steps must be a positive integer.")
        if isinstance(self.featsel_runs, int) and self.featsel_runs <= 0:
            raise ValueError("featsel_runs must be a positive integer.")
        if isinstance(self.max_gb, int) and self.max_gb <= 0:
            raise ValueError("max_gb must be a positive integer.")
        if isinstance(self.n_jobs, int) and self.n_jobs <= 0:
            raise ValueError("n_jobs must be a positive integer.")
        if not isinstance(self.keep_feateng_cols, bool):
            raise TypeError("keep_feateng_cols must be a boolean.")
        if not isinstance(self.pass_feateng_cols, bool):
            raise TypeError("pass_feateng_cols must be a boolean.")
        if self.corr_threshold is not None:
            if not isinstance(self.corr_threshold, float):
                raise TypeError("corr_threshold must be a float.")
            if not 0 <= self.corr_threshold <= 1:
                raise ValueError("corr_threshold must be between 0 and 1.")

    def fit(self, X: pd.DataFrame, y=None) -> "AutoFeatTransformer":
        """
        Fits the AutoFeatTransformer to the input data by fitting the
        AutoFeatRegressor.

        Args:
            X (pd.DataFrame):
                The input data to fit the transformer on.
            y (optional):
                Ignored. Included for compatibility.

        Returns:
            self:
                The fitted transformer instance.

        Raises:
            Exception:
                If there is an error fitting the transformer.
        """
        try:
            self._feateng_cols = X.columns.tolist()
            self._afr.fit(X, y)
            if self.corr_threshold:
                X_transformed = pd.DataFrame(
                    self._afr.transform(X),
                    columns=self._feateng_cols + self._afr.new_feat_cols_,
                )
                self._feature_names = self.find_uncorrelated_features(
                    data=X_transformed,
                    target=y,
                )
            else:
                self._feature_names = self._afr.new_feat_cols_
                if self.pass_feateng_cols:
                    self._feature_names = (
                        self._feateng_cols + self._feature_names
                    )
                    logger.info(
                        f"[{self.__class__.__name__}] [fit] Left "
                        f"{len(self._feature_names)} features out of "
                        f"{len(self._feature_names)}. Input columns were "
                        f"kept. "
                    )
                else:
                    logger.info(
                        f"[{self.__class__.__name__}] [fit] Left "
                        f"{len(self._feature_names)} features out of "
                        f"{len(self._afr.new_feat_cols_) + len(self._feateng_cols)}. Input columns were not "
                        f"kept. "
                    )
            return self
        except Exception as e:
            raise Exception(f"Failed to fit {self.__class__.__name__}") from e

    def find_uncorrelated_features(
        self,
        data: pd.DataFrame,
        target: np.ndarray,
    ) -> List[str]:

        n_input_cols = data.shape[1]
        data["target"] = target
        corr_matrix = data.corr("spearman").abs()
        # corr_matrix = data.phik_matrix()

        target_corr = corr_matrix.loc[:, "target"].sort_values(ascending=False)
        corr_matrix.drop(columns=["target"], index=["target"], inplace=True)
        data.drop(columns=["target"], inplace=True)

        np.fill_diagonal(corr_matrix.values, 0)

        stacked = corr_matrix.stack().sort_values(ascending=False)
        cols_to_drop = set()
        for (col1, col2), corr_value in stacked.items():
            if corr_value >= self.corr_threshold:
                if all([col not in cols_to_drop for col in [col1, col2]]):
                    if self.keep_feateng_cols:
                        if all(
                            [
                                col not in self._feateng_cols
                                for col in [col1, col2]
                            ]
                        ):
                            cols_to_drop.add(
                                col1
                                if target_corr[col1] <= target_corr[col2]
                                else col2
                            )
                        elif col1 not in self._feateng_cols:
                            cols_to_drop.add(col1)
                        elif col2 not in self._feateng_cols:
                            cols_to_drop.add(col2)
                    else:
                        cols_to_drop.add(
                            col1
                            if target_corr[col1] <= target_corr[col2]
                            else col2
                        )
            else:
                break
        if self.pass_feateng_cols and self.keep_feateng_cols:
            logger.info(
                f"[{self.__class__.__name__}] [fit] Left "
                f"{n_input_cols - len(cols_to_drop)} features out of "
                f"{n_input_cols} after correlation filtering. "
                f"Input columns were kept."
            )
        else:
            cols_to_drop = cols_to_drop | set(self._feateng_cols)
            logger.info(
                f"[{self.__class__.__name__}] [fit] Left "
                f"{n_input_cols - len(cols_to_drop)} features out of "
                f"{n_input_cols} after correlation filtering. Input "
                f"columns were not kept."
            )
        return list(set(data.columns) - cols_to_drop)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applyes fitted AutoFeatRegressor to the input data to generate
        new features.

        Args:
            X (pd.DataFrame):
                The input data.

        Returns:
            np.ndarray:
                Array of new features.

        Raises:
            Exception:
                If there is an error applying the transformer.
        """
        try:
            logger.info(
                f"[{self.__class__.__name__}] [transform] Generated "
                f"{len(self._feature_names)} features out of {X.shape[1]}"
            )
            return self._afr.transform(X)[self._feature_names].astype(
                np.float32
            )
        except Exception as e:
            raise Exception(
                f"Failed to apply {self.__class__.__name__}"
            ) from e

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns a list of feature names for the transformed data.

        Returns:
            List[str]:
                A list of the names of the new features and ones that
                were passed through.
        """
        return self._feature_names


class PassthroughTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that passes through or drops
    specified columns from the input data.

    Attributes:
        passthrough_cols (List[str | int] or None):
            A list of column names to pass through.
        drop_cols (List[str | int] or None):
            A list of column names to drop.
        _remaining_cols (List[str | int]):
            A list of column names that are remained as a result.

    Raises:
        TypeError:
            If any of the input arguments are of the wrong type.
        ValueError:
            If any of the input arguments are invalid or inconsistent.
        Exception:
            If there is an error fitting or applying the transformer.
    """

    def __init__(
        self,
        passthrough_cols: List[str | int] | None = None,
        drop_cols: List[str | int] | None = None,
        ignore_cols: List[str | int] | None = None,
        corr_threshold: float | None = None,
    ) -> None:
        """
        Initializes a PassthroughTransformer instance.

        Args:
            passthrough_cols (List[str | int] or None):
                A list of column names to pass through.
                If None and `drop_cols` is None, all columns are
                passed through. If specified with `corr_threshold`,
                these column will be used in correlation filtering, but
                not dropped no matter what.
            drop_cols (List[str | int] or None):
                A list of column names to drop.
            ignore_cols (List[str | int] or None):
                A list of column names to ignore when applying correlation
                filtering. Must not overlap with `drop_cols`.
            corr_threshold (float or None):
                A correlation threshold for feature selection.
                If None, no correlation-based feature selection is performed.
        """
        self.passthrough_cols = (
            passthrough_cols if passthrough_cols != None else []
        )
        self.drop_cols = drop_cols if drop_cols != None else []
        self.ignore_cols = ignore_cols if ignore_cols != None else []
        self.corr_threshold = corr_threshold
        self._validate_input()

    def _validate_input(self) -> None:
        """
        Validates the input arguments

        Raises:
            TypeError:
                If any of the input arguments are of the wrong type.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """
        for name, cols in {
            "passthrough_cols": self.passthrough_cols,
            "drop_cols": self.drop_cols,
            "ignore_cols": self.ignore_cols,
        }.items():
            if cols is not None:
                if not isinstance(cols, list):
                    raise TypeError(f"{name} must be a list of column names.")
                elif any([not isinstance(col, (str, int)) for col in cols]):
                    raise TypeError(f"{name} must be a list of column names.")
                elif len(set(cols)) != len(cols):
                    raise ValueError(
                        f"{name} must be a list of unique column names."
                    )
        if set(self.passthrough_cols) & set(self.drop_cols):
            raise ValueError(
                "passthrough_cols and drop_cols cannot have overlapping columns."
            )
        if set(self.ignore_cols) & set(self.drop_cols):
            raise ValueError(
                "ignore_cols and drop_cols cannot have overlapping columns."
            )
        if self.corr_threshold is not None:
            if not isinstance(self.corr_threshold, float):
                raise TypeError("corr_threshold must be a float.")
            if not (0 <= self.corr_threshold <= 1):
                raise ValueError(
                    "corr_threshold must be a float between 0 and 1."
                )

    def find_uncorrelated_features(
        self,
        data: pd.DataFrame,
        target: np.ndarray,
    ) -> List[str]:

        data["target"] = target
        corr_matrix = data.corr("spearman").abs()
        # corr_matrix = data.phik_matrix()

        target_corr = corr_matrix.loc[:, "target"].sort_values(ascending=False)
        corr_matrix.drop(columns=["target"], index=["target"], inplace=True)
        data.drop(columns=["target"], inplace=True)

        np.fill_diagonal(corr_matrix.values, 0)

        stacked = corr_matrix.stack().sort_values(ascending=False)
        cols_to_drop = set()
        for (col1, col2), corr_value in stacked.items():
            if corr_value >= self.corr_threshold:
                if all([col not in cols_to_drop for col in [col1, col2]]):
                    if all(
                        [
                            col not in self.passthrough_cols
                            for col in [col1, col2]
                        ]
                    ):
                        cols_to_drop.add(
                            col1
                            if target_corr[col1] <= target_corr[col2]
                            else col2
                        )
                    elif col1 not in self.passthrough_cols:
                        cols_to_drop.add(col1)
                    elif col2 not in self.passthrough_cols:
                        cols_to_drop.add(col2)
            else:
                break
        return list(set(data.columns) - set(cols_to_drop))

    def fit(self, X: pd.DataFrame, y=None) -> "PassthroughTransformer":
        """
        Fits the PassthroughTransformer to the input DataFrame `X`.
        The `fit` method determines the columns that will be kept or
        dropped based on the `passthrough_cols` and `drop_cols`
        arguments provided during initialization.

        Args:
            X (pd.DataFrame):
                The input DataFrame to fit the transformer to.
            y (optional):
                Ignored, as this transformer does not use the target
                variable.

        Returns:
            self:
                The fitted transformer.

        Raises:
            Exception:
                If the provided column names are not found in the input
                DataFrame, or if all columns are being dropped.
        """
        fail_msg = f"Failed to fit {self.__class__.__name__}"
        n_input_cols = len(X.columns)
        self._remaining_cols = X.columns.tolist()

        # Validating
        odd_columns = set(self.drop_cols) - set(self._remaining_cols)
        if len(odd_columns) > 0:
            raise Exception(
                f"{fail_msg}. Found drop_cols that are not in the "
                f"input columns: {odd_columns}"
            )
        if len(self.drop_cols) == len(self._remaining_cols):
            raise Exception(
                f"{fail_msg}. Dropping all columns is not allowed."
            )
        odd_columns = set(self.passthrough_cols) - set(self._remaining_cols)
        if len(odd_columns) > 0:
            raise Exception(
                f"{fail_msg}. Found passthrough_cols that are not in the "
                f"input columns: {odd_columns}"
            )
        odd_columns = set(self.ignore_cols) - set(self._remaining_cols)
        if len(odd_columns) > 0:
            raise Exception(
                f"{fail_msg}. Found ignore_cols that are not in the "
                f"input columns: {odd_columns}"
            )

        # Dropping drop_cols
        self._remaining_cols = list(
            set(self._remaining_cols) - set(self.drop_cols)
        )
        if len(self.drop_cols) > 0:
            logger.info(
                f"[{self.__class__.__name__}] [fit] Left "
                f"{len(self._remaining_cols)} features out of "
                f"{n_input_cols} after dropping specified columns."
            )

        # Correlation filtering (without ignore_cols)
        if self.corr_threshold:
            self._remaining_cols = self.find_uncorrelated_features(
                X[list(set(self._remaining_cols) - set(self.ignore_cols))], y
            )
            self._remaining_cols = self.ignore_cols + self._remaining_cols
            logger.info(
                f"[{self.__class__.__name__}] [fit] Left "
                f"{len(self._remaining_cols)} features out of "
                f"{n_input_cols} after correlation filtering."
            )
        # Passing only passthrough_cols if specified without
        # correlation filtering
        elif len(self.passthrough_cols) > 0:
            self._remaining_cols = self.passthrough_cols.copy()
            logger.info(
                f"[{self.__class__.__name__}] [fit] Left "
                f"{len(self._remaining_cols)} features out of "
                f"{n_input_cols} after passing only specified columns."
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input DataFrame `X` by selecting only the
        columns specified in the `_remaining_cols` attribute.

        Args:
            X (pd.DataFrame):
                The input DataFrame to transform.

        Returns:
            np.ndarray:
                Array containing only the columns specified in
                `_remaining_cols`.

        Raises:
            Exception:
                If there is any error during the transformation.
        """
        try:
            logger.info(
                f"[{self.__class__.__name__}] [transform] Passed "
                f"{len(self._remaining_cols)} features out of {X.shape[1]}"
            )
            return X[self._remaining_cols].values
        except Exception as e:
            raise Exception(
                f"Failed to apply {self.__class__.__name__}"
            ) from e

    def get_feature_names_out(self, input_features=None) -> List[str | int]:
        """
        Returns the list of column names that will be included in the
        transformed DataFrame.

        Returns:
            List[str | int]:
                The list of column names that will be included in the
                transformed DataFrame.
        """
        return self._remaining_cols


class FeatureNormaliserTransformer(TransformerMixin, BaseEstimator):
    """
    A custom transformer that normalizes features based on group-wise
    aggregation.

    Attributes:
        features_to_normalise (List[str | int]):
            A list of feature names to be normalized.
        grouping_features (List[str | int]):
            A list of feature names to be used for grouping when
            calculating the aggregation.
        aggregation_method (Literal["mean", "median"], optional):
            The aggregation method, either "mean" or "median".
            Defaults to "mean".
        _agg_values (Dict[str | int, Dict[str | int, float]]):
            A dictionary to store the aggregated values.

    Raises:
        TypeError:
            If any of the input arguments is not of the expected type
        Exception:
            If there is any error during the fitting or transformation.
    """

    def __init__(
        self,
        features_to_normalise: List[str | int],
        grouping_features: List[str | int],
        aggregation_method: Literal["mean", "median"] = "mean",
    ) -> None:
        """
        Initializes a FeatureNormaliserTransformer instance.

        Args:
            features_to_normalise (List[str | int]):
                A list of feature names to be normalized.
            grouping_features (List[str | int]):
                A list of feature names to be used for grouping when
                calculating the aggregation.
            aggregation_method (Literal["mean", "median"], optional):
                The aggregation method, either "mean" or "median".
                Defaults to "mean".
        """
        self.features_to_normalise = features_to_normalise
        self.grouping_features = grouping_features
        self.aggregation_method = aggregation_method
        self._validate_input()

    def _validate_input(self) -> None:
        """
        Validates the input arguments.

        Raises:
            TypeError:
                If any of the input arguments are of the wrong type.
        """
        if not isinstance(self.features_to_normalise, list):
            raise TypeError("features_to_normalise must be a list")
        if not isinstance(self.grouping_features, list):
            raise TypeError("grouping_features must be a list")
        if not all(
            isinstance(x, (str, int)) for x in self.features_to_normalise
        ):
            raise TypeError(
                "features_to_normalise must be a list of strings or integers"
            )
        if not all(isinstance(x, (str, int)) for x in self.grouping_features):
            raise TypeError(
                "grouping_features must be a list of strings or integers"
            )
        if self.aggregation_method not in ["mean", "median"]:
            raise TypeError("aggregation_method must be 'mean' or 'median'")

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureNormaliserTransformer":
        """
        Fits the transformer by calculating the aggregation values for
        each feature and grouping feature combination.

        Args:
            X (pd.DataFrame):
                The input DataFrame to fit the transformer on.
            y (Any, optional):
                Ignored. Included for compatibility.

        Returns:
            self:
                The fitted transformer.

        Raises:
            Exception:
                If there is an error during the fitting process.
        """

        try:
            self._agg_values = {}
            for feature in self.features_to_normalise:
                self._agg_values[feature] = {}
                for col in self.grouping_features:
                    self._agg_values[feature][col] = (
                        X.groupby(col)[feature]
                        .agg(self.aggregation_method)
                        .to_dict()
                    )
            logger.info(
                f"[{self.__class__.__name__}] [fit] Calculated aggregations for "
                f"{len(self.features_to_normalise) * len(self.grouping_features)} "
                f"new features."
            )
            return self

        except Exception as e:
            raise Exception(f"Failed to fit {self.__class__.__name__}") from e

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input DataFrame by normalizing the specified
        features using the aggregation values calculated during the
        fitting process.

        Args:
            X (pd.DataFrame):
                The input DataFrame.

        Returns:
            np.ndarray:
                Array with the normalized features.

        Raises:
            Exception:
                If there is an error during the transformation process.
        """
        try:
            output = []
            for feature in self.features_to_normalise:
                for col in self.grouping_features:
                    output.append(
                        X[[feature, col]]
                        .apply(
                            lambda x: (
                                x[0] - self._agg_values[feature][col][x[1]]
                            )
                            / self._agg_values[feature][col][x[1]],
                            axis=1,
                        )
                        .values.astype(np.float32)
                    )
            logger.info(
                f"[{self.__class__.__name__}] [transform] Calculated "
                f"{len(output)} features out of "
                f"{len(self.features_to_normalise)}."
            )
            return np.array(output).T
        except Exception as e:
            raise Exception(
                f"Failed to apply {self.__class__.__name__}"
            ) from e

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns a list of names for normalized features.

        Returns:
            List[str]:
                A list of feature names.
        """
        return [
            f"{feature}__{col}"
            for feature in self.features_to_normalise
            for col in self.grouping_features
        ]


class DataFrameTransformer(ColumnTransformer):
    """
    A transformer that changes the `ColumnTransformer` as follows:
    - The output is a pandas DataFrame by default and can't be changed
    - Remainder columns are dropped
    - Prefixes are always added to the names of new columns
    - Default `__` separator is removed. Therefore, it is advisable to
      add some separator directly to the prefix
    - modifies `transform` function so that logging can be done
    """

    def __init__(
        self,
        transformers: Sequence[tuple],
    ):
        super().__init__(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self.set_output(transform="pandas")

    def transform(self, *args, **kwargs):

        transformed_data = super().transform(*args, **kwargs)
        logger.info(
            f"[{self.__class__.__name__}] [transform] Generated "
            f"{transformed_data.shape[1]} features out of "
            f"{self.n_features_in_}."
        )
        return transformed_data

    def get_feature_names_out(self, input_features=None):

        return [
            f"{t[0]}{feat}"
            for t in self.transformers_
            for feat in t[1].get_feature_names_out()
        ]


def get_transforming_pipeline(
    transformers: Dict[str, Any], pipeline: Dict[str, Any]
) -> Pipeline:
    """
    Creates a scikit-learn Pipeline object from a dictionary of
    transformers configurations and a dictionary of pipeline steps
    configurations.

    Each step of the pipeline can only be a DataFrameTransformer class
    that inherits from ColumnTransformer.

    Args:
        transformers (Dict[str, Any]):
            A dictionary mapping transformer names to their
            configuration.

        pipeline (Dict[str, Any]):
            A dictionary mapping pipeline step names to their
            configuration, which includes the transformer class
            and parameters.

    Returns:
        Pipeline:
            A scikit-learn Pipeline object that applies the specified
            transformers in the given order.

    Raises:
        Exception:
            If there is an error creating the pipeline of transformers.
    """

    def get_transformer(
        class_name: str,
        prefix: str,
        params: Dict[str, Any],
        columns_selector_params: Dict[str, Any] | None = None,
        columns: List[str] | None = None,
    ) -> Tuple[str, ColumnTransformer, List[str] | Callable]:
        """
        Get a transformer tuple consisting of the transformer prefix,
        the transformer instance, and the columns to apply the
        transformer to (or a callable function to select the columns).

        Args:
            class_name (str):
                The name of the transformer class to instantiate.
            prefix (str):
                The prefix to use for the transformer in the feature
                names.
            params (Dict[str, Any]):
                The parameters to pass to the transformer class
                constructor.
            columns_selector_params (Dict[str, Any] | None, optional):
                The parameters to pass to the `make_column_selector`
                function to select the columns. Defaults to `None`.
            columns (List[str] | None, optional):
                The specific columns to apply the transformer to.
                `None` means that a column selector function will be
                defined to select the columns. Defaults to `None`.

        Returns:
            Tuple[str, ColumnTransformer, List[str] | callable]:
                A tuple containing the transformer prefix, the
                transformer instance, and the columns to apply the
                transformer to (or a callable function to select the
                columns).
        """
        col_params = columns_selector_params if columns_selector_params else {}
        tr_params = params if params else {}
        transformer = globals()[class_name](**tr_params)
        cols = columns if columns else make_column_selector(**col_params)
        return (prefix, transformer, cols)

    def get_column_transformer(
        transformers: List[str],
        transformers_config: Dict[str, Any],
    ) -> DataFrameTransformer:
        """
        Get a ColumnTransformer-based DataFrameTransformer transformer
        instance that applies the specified transformers.

        Args:
            transformers (List[str]):
                The names of the transformers to apply.
            transformers_config (Dict[str, Any]):
                The configuration for each transformer, keyed by the
                transformer name.

        Returns:
            DataFrameTransformer:
                A DataFrameTransformer transformer instance.
        """
        return DataFrameTransformer(
            transformers=[
                get_transformer(**transformers_config[name])
                for name in transformers
            ],
        )

    try:
        return Pipeline(
            steps=[
                (
                    step,
                    get_column_transformer(
                        **config, transformers_config=transformers
                    ),
                )
                for step, config in pipeline.items()
            ]
        )
    except Exception as e:
        msg = f"Failed to create pipeline of transformers"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e
