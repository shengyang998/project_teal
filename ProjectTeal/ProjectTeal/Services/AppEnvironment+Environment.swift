//
//  AppEnvironment+Environment.swift
//  ProjectTeal
//
//  SwiftUI integration helpers for dependency injection.
//

import SwiftUI

private struct AppEnvironmentKey: EnvironmentKey {
    static var defaultValue: AppEnvironment = .shared
}

extension EnvironmentValues {
    var appEnvironment: AppEnvironment {
        get { self[AppEnvironmentKey.self] }
        set { self[AppEnvironmentKey.self] = newValue }
    }
}

extension View {
    /// Injects an AppEnvironment instance into the SwiftUI environment.
    func appEnvironment(_ environment: AppEnvironment) -> some View {
        self.environment(\.appEnvironment, environment)
    }
}
