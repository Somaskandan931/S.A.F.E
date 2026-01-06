"""
Sequence Builder for LSTM Models
Converts zone-level time series into sequences for LSTM training
"""

import numpy as np
import pandas as pd


class SequenceBuilder :
    """Build sequences from time-series data for LSTM"""

    def __init__ ( self, window_size: int = 30, stride: int = 5 ) :
        """
        Initialize sequence builder

        Args:
            window_size: Length of each sequence
            stride: Step size between sequences
        """
        self.window_size = window_size
        self.stride = stride

    def build_sequences ( self, df: pd.DataFrame, feature_columns: list ) -> np.ndarray :
        """
        Build sequences from DataFrame

        Args:
            df: Input DataFrame (should be sorted by time)
            feature_columns: List of feature column names to use

        Returns:
            3D numpy array of shape (num_sequences, window_size, num_features)
        """
        sequences = []

        # Check if zone_id exists
        if 'zone_id' in df.columns :
            # Build sequences per zone
            for zone_id, zone_df in df.groupby( "zone_id" ) :
                zone_df = zone_df.sort_values( "time_window" ).reset_index( drop=True )
                zone_sequences = self._create_sequences_from_zone( zone_df, feature_columns )
                sequences.extend( zone_sequences )
        else :
            # Build sequences from entire dataset (no zone separation)
            print( "⚠️  'zone_id' not found, building sequences from entire dataset" )
            df_sorted = df.sort_values( "time_window" ).reset_index( drop=True )
            sequences = self._create_sequences_from_zone( df_sorted, feature_columns )

        if len( sequences ) == 0 :
            raise ValueError( "No sequences created. Check data length and window_size." )

        return np.array( sequences )

    def _create_sequences_from_zone ( self, zone_df: pd.DataFrame,
                                      feature_columns: list ) -> list :
        """
        Create sequences from a single zone's data

        Args:
            zone_df: DataFrame for one zone
            feature_columns: Feature columns to use

        Returns:
            List of sequences
        """
        sequences = []

        # Extract feature values
        values = zone_df[feature_columns].values

        # Sliding window
        for i in range( 0, len( values ) - self.window_size + 1, self.stride ) :
            sequence = values[i :i + self.window_size]
            sequences.append( sequence )

        return sequences


if __name__ == "__main__" :
    # Test the sequence builder
    print( "Testing SequenceBuilder..." )

    # Create sample data
    data = {
        'zone_id' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'time_window' : [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        'density' : [10, 12, 15, 13, 11, 20, 22, 25, 23, 21],
        'velocity' : [1.0, 1.2, 1.1, 1.3, 1.0, 2.0, 2.1, 2.2, 2.0, 1.9]
    }

    df = pd.DataFrame( data )

    builder = SequenceBuilder( window_size=3, stride=1 )
    sequences = builder.build_sequences( df, ['density', 'velocity'] )

    print( f"Created {len( sequences )} sequences" )
    print( f"Sequence shape: {sequences.shape}" )
    print( f"First sequence:\n{sequences[0]}" )