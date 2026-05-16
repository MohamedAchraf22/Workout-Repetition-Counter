"""Statistics collection and analysis for exercise detection tuning."""

import numpy as np


class NormalizationStatsCollector:
    """
    Collects frame-by-frame angle statistics to analyze thresholds.
    
    Helps identify optimal angle thresholds by recording poses in
    different positions (e.g., TOGETHER vs APART for jump jacks).
    """
    
    def __init__(self):
        """Initialize stats collector."""
        self.arm_together = []
        self.arm_apart = []
        self.leg_together = []
        self.leg_apart = []
        self.frame_data = []
    
    def record_frame(self, arm_angle, leg_angle, position):
        """
        Record a frame's angle measurements.
        
        Args:
            arm_angle: Arm angle in degrees
            leg_angle: Leg angle in degrees
            position: Position label (TOGETHER, APART, TRANSITION, etc.)
        """
        if position == "TOGETHER":
            self.arm_together.append(arm_angle)
            self.leg_together.append(leg_angle)
        elif position == "APART":
            self.arm_apart.append(arm_angle)
            self.leg_apart.append(leg_angle)
        
        self.frame_data.append(
            {"arm_angle": arm_angle, "leg_angle": leg_angle, "position": position}
        )
    
    def _compute_stats(self, values):
        """Compute statistics for a list of values."""
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "count": len(values),
        }
    
    def _suggest_threshold(self, together_stats, apart_stats):
        """Suggest optimal threshold between two groups."""
        if not together_stats or not apart_stats:
            return None
        
        together_max = together_stats["max"]
        apart_min = apart_stats["min"]
        if apart_min > together_max:
            threshold = (together_max + apart_min) / 2
            overlap = False
        else:
            threshold = (together_stats["mean"] + apart_stats["mean"]) / 2
            overlap = True
        
        return threshold, overlap
    
    def _text_visualization(self, together_stats, apart_stats, threshold, metric_name):
        """Create ASCII visualization of angle ranges."""
        if not together_stats or not apart_stats:
            return ""
        
        together_max = together_stats["max"]
        apart_min = apart_stats["min"]
        apart_max = apart_stats["max"]
        together_min = together_stats["min"]
        
        min_val = min(together_min, apart_min)
        max_val = max(together_max, apart_max)
        range_val = max_val - min_val if max_val > min_val else 1
        scale = lambda v: int(((v - min_val) / range_val) * 50)
        together_max_pos = scale(together_max)
        apart_min_pos = scale(apart_min)
        threshold_pos = scale(threshold)
        apart_max_pos = scale(apart_max)
        line = ""
        for i in range(52):
            if i == threshold_pos:
                line += "|"
            elif together_max_pos >= i >= 0:
                line += "="
            elif apart_min_pos <= i <= apart_max_pos:
                line += "="
            else:
                line += " "
        
        return f"{metric_name:15s} TOGETHER {line} APART"
    
    def analyze_and_print(self):
        """Print comprehensive analysis of collected statistics."""
        print("\n" + "=" * 80)
        print("JUMP JACK ANGLE-BASED DETECTION ANALYSIS")
        print("=" * 80 + "\n")
        
        if not self.frame_data:
            print("⚠️  No frame data collected!")
            return
        
        # Compute statistics
        arm_together_stats = self._compute_stats(self.arm_together)
        arm_apart_stats = self._compute_stats(self.arm_apart)
        leg_together_stats = self._compute_stats(self.leg_together)
        leg_apart_stats = self._compute_stats(self.leg_apart)
        
        # ARM ANGLE ANALYSIS
        print("ARM ANGLE ANALYSIS (elbow → shoulder → hip)")
        print("-" * 80)
        
        if arm_together_stats:
            print(
                f"  TOGETHER: min={arm_together_stats['min']:.1f}°, "
                f"max={arm_together_stats['max']:.1f}°, "
                f"mean={arm_together_stats['mean']:.1f}° ± {arm_together_stats['std']:.1f}° "
                f"(n={arm_together_stats['count']})"
            )
        else:
            print("  TOGETHER: No data collected")
        
        if arm_apart_stats:
            print(
                f"  APART:    min={arm_apart_stats['min']:.1f}°, "
                f"max={arm_apart_stats['max']:.1f}°, "
                f"mean={arm_apart_stats['mean']:.1f}° ± {arm_apart_stats['std']:.1f}° "
                f"(n={arm_apart_stats['count']})"
            )
        else:
            print("  APART:    No data collected")
        
        arm_threshold, arm_overlap = self._suggest_threshold(
            arm_together_stats, arm_apart_stats
        )
        
        if arm_threshold is not None:
            if arm_overlap:
                print(
                    f"\n  ⚠️  WARNING: OVERLAP DETECTED between TOGETHER and APART ranges!"
                )
                print(
                    f"      Consider using hysteresis filtering with separate open/closed thresholds"
                )
            else:
                print(f"\n  ✓ Clear separation detected")
            print(f"  🎯 SUGGESTED THRESHOLD (midpoint): {arm_threshold:.1f}°")
        
        # Visualization
        if arm_together_stats and arm_apart_stats:
            viz_line = self._text_visualization(
                arm_together_stats, arm_apart_stats, arm_threshold, "ARM ANGLE"
            )
            print(f"\n  {viz_line}")
        
        print()
        
        # LEG ANGLE ANALYSIS
        print("LEG ANGLE ANALYSIS (knee → hip → opposite hip)")
        print("-" * 80)
        
        if leg_together_stats:
            print(
                f"  TOGETHER: min={leg_together_stats['min']:.1f}°, "
                f"max={leg_together_stats['max']:.1f}°, "
                f"mean={leg_together_stats['mean']:.1f}° ± {leg_together_stats['std']:.1f}° "
                f"(n={leg_together_stats['count']})"
            )
        else:
            print("  TOGETHER: No data collected")
        
        if leg_apart_stats:
            print(
                f"  APART:    min={leg_apart_stats['min']:.1f}°, "
                f"max={leg_apart_stats['max']:.1f}°, "
                f"mean={leg_apart_stats['mean']:.1f}° ± {leg_apart_stats['std']:.1f}° "
                f"(n={leg_apart_stats['count']})"
            )
        else:
            print("  APART:    No data collected")
        
        leg_threshold, leg_overlap = self._suggest_threshold(
            leg_together_stats, leg_apart_stats
        )
        
        if leg_threshold is not None:
            if leg_overlap:
                print(
                    f"\n  ⚠️  WARNING: OVERLAP DETECTED between TOGETHER and APART ranges!"
                )
                print(
                    f"      Consider using hysteresis filtering with separate open/closed thresholds"
                )
            else:
                print(f"\n  ✓ Clear separation detected")
            print(f"  🎯 SUGGESTED THRESHOLD (midpoint): {leg_threshold:.1f}°")
        
        # Visualization
        if leg_together_stats and leg_apart_stats:
            viz_line = self._text_visualization(
                leg_together_stats, leg_apart_stats, leg_threshold, "LEG ANGLE"
            )
            print(f"\n  {viz_line}")
        
        print()
        
        # RECOMMENDATIONS
        print("RECOMMENDATIONS")
        print("-" * 80)
        
        # Determine stability
        arm_stability = (
            arm_together_stats["std"] + arm_apart_stats["std"]
            if (arm_together_stats and arm_apart_stats)
            else float("inf")
        )
        leg_stability = (
            leg_together_stats["std"] + leg_apart_stats["std"]
            if (leg_together_stats and leg_apart_stats)
            else float("inf")
        )
        
        if arm_stability < leg_stability:
            print(
                f"  📊 ARM angle is more stable (total std: {arm_stability:.2f}° vs leg: {leg_stability:.2f}°)"
            )
            print(f"     Arm thresholds will be more reliable")
        elif leg_stability < arm_stability:
            print(
                f"  📊 LEG angle is more stable (total std: {leg_stability:.2f}° vs arm: {arm_stability:.2f}°)"
            )
            print(f"     Leg thresholds will be more reliable")
        else:
            print(f"  📊 Both angles have similar stability")
        
        print(f"\n  📝 Current configuration:")
        print(f"     ARM_OPEN_THRESHOLD:   130°  (suggested: {arm_threshold:.1f}°)")
        print(f"     ARM_CLOSED_THRESHOLD:  80°  (can be tuned lower for hysteresis)")
        print(f"     LEG_OPEN_THRESHOLD:   100°  (suggested: {leg_threshold:.1f}°)")
        print(f"     LEG_CLOSED_THRESHOLD:  50°  (can be tuned lower for hysteresis)")
        
        # Decision support for incomplete data
        print("\n")
        self._provide_decision_support(
            arm_together_stats, arm_apart_stats, leg_together_stats, leg_apart_stats
        )
        
        print("\n" + "=" * 80 + "\n")
    
    def _provide_decision_support(
        self, arm_together_stats, arm_apart_stats, leg_together_stats, leg_apart_stats
    ):
        """Provide guidance for threshold tuning based on collected data."""
        print("THRESHOLD DECISION SUPPORT")
        print("-" * 80)
        
        # Check for missing categories
        missing_together = (
            not arm_together_stats or arm_together_stats["count"] == 0
        ) and (not leg_together_stats or leg_together_stats["count"] == 0)
        missing_apart = (
            not arm_apart_stats or arm_apart_stats["count"] == 0
        ) and (not leg_apart_stats or leg_apart_stats["count"] == 0)
        
        if missing_together:
            print("  ⚠️  NO TOGETHER DATA DETECTED")
            print("     This means your person never returned to the starting position during video.")
            print("     Possible causes:")
            print("       1. Video too short - didn't complete full jump jacks")
            print("       2. Person started with arms/legs apart")
            print("       3. Thresholds too high - filters out TOGETHER positions")
            print("\n     ACTION: Record a longer video including start and end positions")
            
            # Provide threshold suggestions based on APART data
            if arm_apart_stats:
                print(f"\n  💡 ARM Threshold Suggestion (based on APART data only):")
                print(f"     Current: ARM_OPEN_THRESHOLD=130°")
                print(
                    f"     APART angle range: {arm_apart_stats['min']:.1f}° to {arm_apart_stats['max']:.1f}°"
                )
                print(f"     Try setting ARM_OPEN_THRESHOLD BELOW APART minimum:")
                print(
                    f"       - Conservative: {arm_apart_stats['min'] * 0.9:.1f}° (90% of min)"
                )
                print(
                    f"       - Moderate:     {arm_apart_stats['min'] * 0.95:.1f}° (95% of min)"
                )
            
            if leg_apart_stats:
                print(f"\n  💡 LEG Threshold Suggestion (based on APART data only):")
                print(f"     Current: LEG_OPEN_THRESHOLD=100°")
                print(
                    f"     APART angle range: {leg_apart_stats['min']:.1f}° to {leg_apart_stats['max']:.1f}°"
                )
                print(f"     Try setting LEG_OPEN_THRESHOLD BELOW APART minimum:")
                print(
                    f"       - Conservative: {leg_apart_stats['min'] * 0.9:.1f}° (90% of min)"
                )
                print(
                    f"       - Moderate:     {leg_apart_stats['min'] * 0.95:.1f}° (95% of min)"
                )
        
        elif missing_apart:
            print("  ⚠️  NO APART DATA DETECTED")
            print("     This means your person never spread arms/legs during video.")
            print("     Possible causes:")
            print("       1. Video only shows static position")
            print("       2. Thresholds too high - prevents APART detection")
            print("       3. Person didn't perform full range motion")
            print("\n     ACTION: Record a video with full arm/leg spread during motion")
        
        else:
            print("  ✓ Both TOGETHER and APART data present")
            print("    Analysis above should be reliable")
        
        print("\n  📋 THRESHOLD TUNING STRATEGY:")
        print("     1. Start with suggested thresholds from TOGETHER/APART ranges")
        print("     2. Use OPEN thresholds for max APART angle, CLOSED for min TOGETHER")
        print("     3. Test with a 10-second video clip")
        print("     4. Monitor debug output showing average arm/leg angles")
        print("     5. Adjust thresholds ±5° if needed")
        print("     6. Look for:")
        print("        - TOGETHER position should have SMALL angles")
        print("        - APART position should have LARGE angles")
        print("        - Angles should transition clearly between positions")
